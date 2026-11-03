// Engine/Chapter 07：真实 TinyLlama sequential 与 scalar Linear batch A/B。
// 职责：固定双请求 decode 负载，记录逐轮墙钟、吞吐与配对加速比。
// 运行：real_tinyllama_batch_benchmark <model.gguf> [--warmup N] [--iterations N] [--output PATH]。
#include "chapter06_generation/sampler.h"
#include "cpp_llm_engine/real_llama_executor_factory.h"
#include "cpp_llm_engine/real_tokenizer_backend.h"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;
using Executor = chapter06_generation::RequestModelExecutor;
using ExecutorList = std::vector<std::shared_ptr<Executor>>;
using LogitsBatch = std::vector<chapter06_generation::Logits>;
using TokenSequence = chapter06_generation::TokenSequence;
using DecoderProfile = cpp_llm_engine::RealLlamaExecutorProfile;

#ifndef CPP_LLM_ENGINE_BUILD_TYPE
#define CPP_LLM_ENGINE_BUILD_TYPE "unknown"
#endif

double elapsed_ms(Clock::time_point start, Clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

std::string cpu_model_name() {
#if defined(__linux__)
    std::ifstream cpuinfo("/proc/cpuinfo");
    std::string line;
    while (std::getline(cpuinfo, line)) {
        const std::string key = "model name";
        if (line.compare(0, key.size(), key) != 0) continue;
        const std::size_t separator = line.find(':');
        const std::size_t first = line.find_first_not_of(" \t", separator + 1);
        return first == std::string::npos ? "unknown" : line.substr(first);
    }
#endif
    return "unknown";
}

std::string compiler_name() {
#if defined(__clang__)
    return std::string("clang ") + __clang_version__;
#elif defined(__GNUC__)
    return std::string("gcc ") + __VERSION__;
#else
    return "unknown";
#endif
}

std::string json_escape(const std::string& value) {
    std::string escaped;
    for (const char character : value) {
        if (character == '\\' || character == '"') escaped.push_back('\\');
        escaped.push_back(character);
    }
    return escaped;
}

struct ModeResult {
    double decode_ms = 0.0;
    std::vector<TokenSequence> generated;
    DecoderProfile profile;
    bool correctness = false;
};

struct PairResult {
    ModeResult sequential;
    ModeResult batch;
    double speedup = 0.0;
    bool correctness = false;
};

struct Summary {
    double minimum = 0.0;
    double mean = 0.0;
    double median = 0.0;
    double maximum = 0.0;
};

Summary summarize(std::vector<double> values) {
    if (values.empty()) throw std::invalid_argument("benchmark summary requires samples");
    std::sort(values.begin(), values.end());
    const double sum = std::accumulate(values.begin(), values.end(), 0.0);
    const std::size_t middle = values.size() / 2;
    const double median = values.size() % 2 == 0
        ? (values[middle - 1] + values[middle]) / 2.0 : values[middle];
    return {values.front(), sum / static_cast<double>(values.size()),
            median, values.back()};
}

bool prepare(const std::vector<TokenSequence>& prompts,
             cpp_llm_engine::RealLlamaExecutorFactory& factory,
             ExecutorList* executors,
             LogitsBatch* logits,
             std::string* error) {
    executors->clear();
    logits->assign(prompts.size(), {});
    for (std::size_t index = 0; index < prompts.size(); ++index) {
        auto executor = factory.create_executor();
        if (executor == nullptr ||
            !executor->prefill(prompts[index], &(*logits)[index], error)) {
            return false;
        }
        executors->push_back(std::move(executor));
    }
    return true;
}

void add_profile(DecoderProfile* target, const DecoderProfile& source) {
    target->forward_tokens += source.forward_tokens;
    target->total_ms += source.total_ms;
    target->embedding_ms += source.embedding_ms;
    target->normalization_ms += source.normalization_ms;
    target->qkv_projection_ms += source.qkv_projection_ms;
    target->rope_ms += source.rope_ms;
    target->attention_ms += source.attention_ms;
    target->attention_output_ms += source.attention_output_ms;
    target->ffn_ms += source.ffn_ms;
    target->ffn_gate_ms += source.ffn_gate_ms;
    target->ffn_up_ms += source.ffn_up_ms;
    target->ffn_activation_ms += source.ffn_activation_ms;
    target->ffn_down_ms += source.ffn_down_ms;
    target->ffn_residual_ms += source.ffn_residual_ms;
    target->lm_head_ms += source.lm_head_ms;
    target->batch_linear_calls += source.batch_linear_calls;
    target->batch_input_pack_ms += source.batch_input_pack_ms;
    target->batch_kernel_ms += source.batch_kernel_ms;
    target->batch_output_unpack_ms += source.batch_output_unpack_ms;
}

DecoderProfile collect_profile(
    bool use_batch,
    const ExecutorList& executors,
    cpp_llm_engine::RealLlamaExecutorFactory& factory) {
    std::vector<DecoderProfile> profiles(executors.size());
    std::string error;
    for (std::size_t index = 0; index < executors.size(); ++index) {
        if (!factory.executor_profile(executors[index], &profiles[index], &error)) {
            throw std::runtime_error("read Decoder profile failed: " + error);
        }
    }
    if (!use_batch) {
        DecoderProfile result;
        for (const auto& profile : profiles) add_profile(&result, profile);
        return result;
    }

    // Batch 公共阶段会写入每个 Context；只取一份，再汇总逐请求 embedding/token。
    DecoderProfile result = profiles.front();
    result.forward_tokens = 0;
    result.embedding_ms = 0.0;
    for (const auto& profile : profiles) {
        result.forward_tokens += profile.forward_tokens;
        result.embedding_ms += profile.embedding_ms;
    }
    return result;
}

ModeResult run_mode(bool use_batch,
                    const std::vector<TokenSequence>& prompts,
                    cpp_llm_engine::RealLlamaExecutorFactory& factory) {
    ExecutorList executors;
    LogitsBatch logits;
    std::string error;
    if (!prepare(prompts, factory, &executors, &logits, &error)) {
        throw std::runtime_error("benchmark prefill failed: " + error);
    }
    for (const auto& executor : executors) {
        if (!factory.set_executor_profiling(executor, true, &error)) {
            throw std::runtime_error("enable Decoder profiling failed: " + error);
        }
    }

    chapter06_generation::GreedySampler sampler;
    std::vector<TokenSequence> generated(prompts.size());
    constexpr std::size_t kNewTokens = 4;
    const auto start = Clock::now();
    for (std::size_t step = 0; step < kNewTokens; ++step) {
        TokenSequence next(executors.size());
        for (std::size_t index = 0; index < executors.size(); ++index) {
            next[index] = sampler.sample(logits[index]);
            generated[index].push_back(next[index]);
        }
        if (step + 1 == kNewTokens) break;

        if (use_batch) {
            std::vector<std::size_t> positions(executors.size());
            for (std::size_t index = 0; index < executors.size(); ++index) {
                positions[index] = executors[index]->execution_state().committed_tokens;
            }
            if (!factory.decode_batch(executors, next, positions, &logits, &error)) {
                throw std::runtime_error("batch decode failed: " + error);
            }
        } else {
            for (std::size_t index = 0; index < executors.size(); ++index) {
                if (!executors[index]->decode(next[index], &logits[index], &error)) {
                    throw std::runtime_error("sequential decode failed: " + error);
                }
            }
        }
    }
    const double decode_ms = elapsed_ms(start, Clock::now());
    const std::vector<TokenSequence> expected{
        {29892, 306, 29915, 29885},
        {839, 744, 29892, 263},
    };
    const bool correctness = generated == expected;
    return {
        decode_ms,
        std::move(generated),
        collect_profile(use_batch, executors, factory),
        correctness,
    };
}

PairResult run_pair(std::size_t iteration,
                    const std::vector<TokenSequence>& prompts,
                    cpp_llm_engine::RealLlamaExecutorFactory& factory) {
    PairResult result;
    if (iteration % 2 == 0) {
        result.sequential = run_mode(false, prompts, factory);
        result.batch = run_mode(true, prompts, factory);
    } else {
        result.batch = run_mode(true, prompts, factory);
        result.sequential = run_mode(false, prompts, factory);
    }
    result.speedup = result.batch.decode_ms > 0.0
        ? result.sequential.decode_ms / result.batch.decode_ms : 0.0;
    result.correctness = result.sequential.correctness && result.batch.correctness &&
        result.sequential.generated == result.batch.generated;
    return result;
}

void write_summary(std::ostream& output, const Summary& value) {
    output << "{\"min\": " << value.minimum << ", \"mean\": " << value.mean
           << ", \"median\": " << value.median << ", \"max\": "
           << value.maximum << "}";
}

void write_profile(std::ostream& output, const DecoderProfile& profile) {
    output << "{\"forward_tokens\": " << profile.forward_tokens
           << ", \"total_ms\": " << profile.total_ms
           << ", \"embedding_ms\": " << profile.embedding_ms
           << ", \"normalization_ms\": " << profile.normalization_ms
           << ", \"qkv_projection_ms\": " << profile.qkv_projection_ms
           << ", \"rope_ms\": " << profile.rope_ms
           << ", \"attention_ms\": " << profile.attention_ms
           << ", \"attention_output_ms\": " << profile.attention_output_ms
           << ", \"ffn_ms\": " << profile.ffn_ms
           << ", \"lm_head_ms\": " << profile.lm_head_ms
           << ", \"batch_linear_calls\": " << profile.batch_linear_calls
           << ", \"batch_input_pack_ms\": " << profile.batch_input_pack_ms
           << ", \"batch_kernel_ms\": " << profile.batch_kernel_ms
           << ", \"batch_output_unpack_ms\": "
           << profile.batch_output_unpack_ms << "}";
}

void write_json(const std::string& path,
                std::size_t warmup,
                const std::vector<TokenSequence>& prompts,
                const std::vector<PairResult>& results) {
    std::ofstream output(path);
    if (!output) throw std::runtime_error("cannot open benchmark result: " + path);
    std::vector<double> sequential_ms;
    std::vector<double> batch_ms;
    std::vector<double> speedups;
    bool correctness = true;
    for (const auto& result : results) {
        sequential_ms.push_back(result.sequential.decode_ms);
        batch_ms.push_back(result.batch.decode_ms);
        speedups.push_back(result.speedup);
        correctness = correctness && result.correctness;
    }
    const auto sequential_summary = summarize(sequential_ms);
    const auto batch_summary = summarize(batch_ms);
    const auto speedup_summary = summarize(speedups);
    output << std::setprecision(10)
           << "{\n"
           << "  \"schema_version\": 2,\n"
           << "  \"experiment_id\": \"EXP-CPU-10\",\n"
           << "  \"workload_id\": \"WL-CPU-REAL-BATCH-V1\",\n"
           << "  \"benchmark\": \"tinyllama_real_decode_batch_cpu\",\n"
           << "  \"model\": \"TinyLlama-1.1B-Chat-v1.0.Q4_K_M\",\n"
           << "  \"device\": \"cpu\",\n"
           << "  \"execution_policy\": \"scalar\",\n"
           << "  \"workload\": {\"request_count\": 2, \"prompt_tokens\": ["
           << prompts[0].size() << ", " << prompts[1].size()
           << "], \"decode_calls_per_request\": 3, \"generated_tokens_per_request\": 4},\n"
           << "  \"measurement_scope\": \"decode_only_excludes_model_load_context_creation_and_prefill\",\n"
           << "  \"environment\": {\"hardware_concurrency\": "
           << std::thread::hardware_concurrency() << ", \"cpu_model\": \""
           << json_escape(cpu_model_name()) << "\", \"compiler\": \""
           << json_escape(compiler_name()) << "\", \"build_type\": \""
           << CPP_LLM_ENGINE_BUILD_TYPE << "\"},\n"
           << "  \"warmup_pairs\": " << warmup << ",\n"
           << "  \"iterations\": " << results.size() << ",\n"
           << "  \"samples\": [\n";
    for (std::size_t index = 0; index < results.size(); ++index) {
        output << "    {\"iteration\": " << index
               << ", \"order\": \"" << (index % 2 == 0 ? "sequential-batch" : "batch-sequential")
               << "\", \"sequential_decode_ms\": " << results[index].sequential.decode_ms
               << ", \"batch_decode_ms\": " << results[index].batch.decode_ms
               << ", \"speedup\": " << results[index].speedup
               << ", \"correctness\": " << (results[index].correctness ? "true" : "false")
               << ", \"sequential_profile\": ";
        write_profile(output, results[index].sequential.profile);
        output << ", \"batch_profile\": ";
        write_profile(output, results[index].batch.profile);
        output << "}" << (index + 1 == results.size() ? "\n" : ",\n");
    }
    output << "  ],\n  \"summary\": {\n    \"sequential_decode_ms\": ";
    write_summary(output, sequential_summary);
    output << ",\n    \"batch_decode_ms\": ";
    write_summary(output, batch_summary);
    output << ",\n    \"paired_speedup\": ";
    write_summary(output, speedup_summary);
    output << "\n  },\n"
           << "  \"correctness\": {\"all_iterations\": "
           << (correctness ? "true" : "false")
           << ", \"sequential_batch_tokens_match\": "
           << (correctness ? "true" : "false")
           << ", \"golden_tokens_match\": " << (correctness ? "true" : "false")
           << "}\n}\n";
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc < 2) {
            std::cerr << "usage: real_tinyllama_batch_benchmark <model.gguf> "
                         "[--warmup N] [--iterations N] [--output PATH]\n";
            return 2;
        }
        std::size_t warmup = 1;
        std::size_t iterations = 3;
        std::string output_path = "tinyllama_real_batch_scalar.json";
        for (int index = 2; index < argc; ++index) {
            const std::string argument = argv[index];
            if (argument == "--warmup" && index + 1 < argc) {
                warmup = static_cast<std::size_t>(std::stoul(argv[++index]));
            } else if (argument == "--iterations" && index + 1 < argc) {
                iterations = static_cast<std::size_t>(std::stoul(argv[++index]));
            } else if (argument == "--output" && index + 1 < argc) {
                output_path = argv[++index];
            } else {
                std::cerr << "invalid benchmark argument: " << argument << '\n';
                return 2;
            }
        }
        if (iterations == 0) {
            std::cerr << "iterations must be greater than zero\n";
            return 2;
        }

        std::string error;
        auto tokenizer = cpp_llm_engine::RealTokenizerBackend::load(argv[1], &error);
        auto factory = cpp_llm_engine::RealLlamaExecutorFactory::load(argv[1], &error);
        if (tokenizer == nullptr || factory == nullptr) {
            std::cerr << "benchmark load failed: " << error << '\n';
            return 1;
        }
        const std::vector<TokenSequence> prompts{
            tokenizer->encode("Hello TinyLlama"), tokenizer->encode("Hi")};
        for (std::size_t index = 0; index < warmup; ++index) {
            if (!run_pair(index, prompts, *factory).correctness) {
                std::cerr << "warmup correctness failed\n";
                return 1;
            }
        }
        std::vector<PairResult> results;
        results.reserve(iterations);
        bool correctness = true;
        for (std::size_t index = 0; index < iterations; ++index) {
            results.push_back(run_pair(index, prompts, *factory));
            correctness = correctness && results.back().correctness;
        }
        write_json(output_path, warmup, prompts, results);

        std::vector<double> sequential_ms;
        std::vector<double> batch_ms;
        std::vector<double> speedups;
        for (const auto& result : results) {
            sequential_ms.push_back(result.sequential.decode_ms);
            batch_ms.push_back(result.batch.decode_ms);
            speedups.push_back(result.speedup);
        }
        std::cout << std::setprecision(6)
                  << "benchmark=tinyllama_real_decode_batch_cpu\n"
                  << "correctness=" << (correctness ? "PASS" : "FAIL") << '\n'
                  << "iterations=" << iterations << '\n'
                  << "sequential_decode_mean_ms=" << summarize(sequential_ms).mean << '\n'
                  << "batch_decode_mean_ms=" << summarize(batch_ms).mean << '\n'
                  << "paired_speedup_mean=" << summarize(speedups).mean << '\n'
                  << "result_file=" << output_path << '\n';
        return correctness ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "benchmark_error=" << error.what() << '\n';
        return 1;
    }
}
