// Engine/Chapter 09：真实 TinyLlama 双请求 ServiceCore benchmark。
// 职责：记录模型加载、prefill/decode、TTFT、完成时间、吞吐和正确性证据。
// 运行：real_tinyllama_serving_benchmark <model.gguf> [--warmup N] [--iterations N] [--output PATH]。
#include "cpp_llm_engine/real_llama_executor_factory.h"
#include "cpp_llm_engine/real_tokenizer_backend.h"
#include "chapter06_generation/request.h"
#include "chapter06_generation/runtime.h"
#include "chapter06_generation/sampler.h"
#include "chapter09_serving/serving_core.h"

#include <algorithm>
#include <array>
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
#include <utility>
#include <vector>

#if defined(__linux__) || defined(__APPLE__)
#include <sys/resource.h>
#endif

namespace {

using Clock = std::chrono::steady_clock;
using chapter06_generation::Logits;
using chapter06_generation::RequestModelExecutor;
using chapter06_generation::TokenId;
using chapter06_generation::TokenSequence;

#ifndef CPP_LLM_ENGINE_BUILD_TYPE
#define CPP_LLM_ENGINE_BUILD_TYPE "unknown"
#endif

double elapsed_ms(Clock::time_point start, Clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

std::size_t peak_rss_bytes() {
#if defined(__linux__) || defined(__APPLE__)
    rusage usage{};
    if (getrusage(RUSAGE_SELF, &usage) != 0) return 0;
#if defined(__APPLE__)
    return static_cast<std::size_t>(usage.ru_maxrss);
#else
    return static_cast<std::size_t>(usage.ru_maxrss) * 1024;
#endif
#else
    return 0;
#endif
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
    escaped.reserve(value.size());
    for (const char character : value) {
        if (character == '\\' || character == '"') escaped.push_back('\\');
        escaped.push_back(character);
    }
    return escaped;
}

struct ExecutorTiming {
    double prefill_ms = 0.0;
    double decode_ms = 0.0;
    std::size_t prefill_calls = 0;
    std::size_t decode_calls = 0;
};

class TimedExecutor final : public RequestModelExecutor {
public:
    TimedExecutor(std::shared_ptr<RequestModelExecutor> inner, ExecutorTiming* timing)
        : inner_(std::move(inner)), timing_(timing) {}

    bool prefill(const TokenSequence& prompt, Logits* logits, std::string* error) override {
        const auto start = Clock::now();
        const bool ok = inner_->prefill(prompt, logits, error);
        timing_->prefill_ms += elapsed_ms(start, Clock::now());
        ++timing_->prefill_calls;
        return ok;
    }

    bool supports_chunked_prefill() const override {
        return inner_->supports_chunked_prefill();
    }

    bool prefill_chunk(const TokenSequence& prompt,
                       bool start_new_sequence,
                       bool produce_logits,
                       Logits* logits,
                       std::string* error) override {
        const auto start = Clock::now();
        const bool ok = inner_->prefill_chunk(
            prompt, start_new_sequence, produce_logits, logits, error);
        timing_->prefill_ms += elapsed_ms(start, Clock::now());
        ++timing_->prefill_calls;
        return ok;
    }

    bool decode(TokenId token, Logits* logits, std::string* error) override {
        const auto start = Clock::now();
        const bool ok = inner_->decode(token, logits, error);
        timing_->decode_ms += elapsed_ms(start, Clock::now());
        ++timing_->decode_calls;
        return ok;
    }

    chapter06_generation::ModelExecutionState execution_state() const override {
        return inner_->execution_state();
    }

    const std::shared_ptr<RequestModelExecutor>& inner() const { return inner_; }

private:
    std::shared_ptr<RequestModelExecutor> inner_;
    ExecutorTiming* timing_ = nullptr;
};

struct RunMeasurement {
    double prefill_compute_ms = 0.0;
    double decode_compute_ms = 0.0;
    double runtime_overhead_ms = 0.0;
    double overlapped_compute_ms = 0.0;
    double dispatch_ms = 0.0;
    double end_to_end_ms = 0.0;
    double throughput_tokens_per_second = 0.0;
    std::array<double, 2> ttft_ms{};
    std::array<double, 2> completion_ms{};
    std::array<std::size_t, 2> committed_tokens{};
    std::array<std::size_t, 2> kv_cache_bytes{};
    bool correctness = false;
};

struct Summary {
    double minimum = 0.0;
    double mean = 0.0;
    double p50 = 0.0;
    double maximum = 0.0;
};

Summary summarize(std::vector<double> values) {
    if (values.empty()) throw std::invalid_argument("benchmark summary requires samples");
    std::sort(values.begin(), values.end());
    const double sum = std::accumulate(values.begin(), values.end(), 0.0);
    const std::size_t middle = values.size() / 2;
    const double p50 = values.size() % 2 == 0
        ? (values[middle - 1] + values[middle]) / 2.0
        : values[middle];
    return {values.front(), sum / static_cast<double>(values.size()), p50, values.back()};
}

RunMeasurement run_once(
    const TokenSequence& prompt_a,
    const TokenSequence& prompt_b,
    cpp_llm_engine::RealLlamaExecutorFactory& factory,
    bool parallel_requests) {
    constexpr std::size_t request_a_id = 101;
    constexpr std::size_t request_b_id = 102;
    constexpr std::size_t max_new_tokens = 4;
    const TokenSequence expected_a{29892, 306, 29915, 29885};
    const TokenSequence expected_b{839, 744, 29892, 263};

    const auto end_to_end_start = Clock::now();
    chapter06_generation::GenerationConfig config;
    config.max_new_tokens = max_new_tokens;
    config.eos_token_id = factory.eos_token_id();
    chapter06_generation::Runtime runtime(config);
    chapter06_generation::GreedySampler sampler;
    chapter06_generation::Request request_a(prompt_a);
    chapter06_generation::Request request_b(prompt_b);
    ExecutorTiming timing_a;
    ExecutorTiming timing_b;

    auto inner_a = factory.create_executor();
    auto inner_b = factory.create_executor();
    if (inner_a == nullptr || inner_b == nullptr) return {};
    auto executor_a = std::make_shared<TimedExecutor>(std::move(inner_a), &timing_a);
    auto executor_b = std::make_shared<TimedExecutor>(std::move(inner_b), &timing_b);
    std::weak_ptr<TimedExecutor> weak_a = executor_a;
    std::weak_ptr<TimedExecutor> weak_b = executor_b;

    chapter09_serving::ServingCore service(2, 2);
    const bool admitted_a = service.admit(
        request_a_id, request_a, sampler, std::move(executor_a)).accepted();
    const bool admitted_b = service.admit(
        request_b_id, request_b, sampler, std::move(executor_b)).accepted();

    std::array<double, 2> ttft_ms{};
    std::array<double, 2> completion_ms{};
    const auto dispatch_start = Clock::now();
    chapter09_serving::ServiceStats stats;
    if (parallel_requests) {
        stats = service.dispatch_parallel_requests(
            runtime,
            2,
            [&](const chapter09_serving::ParallelRequestObservation& observation) {
                const std::size_t index = observation.request_id == request_a_id ? 0 : 1;
                ttft_ms[index] = observation.ttft_seconds * 1000.0;
                completion_ms[index] = observation.completion_seconds * 1000.0;
            });
    } else {
        stats = service.dispatch(
            runtime,
            [&](std::size_t request_id, TokenId, std::size_t step) {
                const std::size_t index = request_id == request_a_id ? 0 : 1;
                const double current_ms = elapsed_ms(dispatch_start, Clock::now());
                if (step == 0) ttft_ms[index] = current_ms;
                if (step + 1 == max_new_tokens) completion_ms[index] = current_ms;
            });
    }
    const auto dispatch_end = Clock::now();

    const bool cache_isolated =
        request_a.cache().size() == prompt_a.size() + expected_a.size() &&
        request_b.cache().size() == prompt_b.size() + expected_b.size();
    std::array<std::size_t, 2> committed_tokens{};
    std::array<std::size_t, 2> kv_cache_bytes{};
    bool memory_observed = false;
    auto observed_a = weak_a.lock();
    auto observed_b = weak_b.lock();
    if (observed_a != nullptr && observed_b != nullptr) {
        cpp_llm_engine::RealLlamaExecutorMemory memory_a;
        cpp_llm_engine::RealLlamaExecutorMemory memory_b;
        std::string memory_error;
        memory_observed = factory.executor_memory(
            observed_a->inner(), &memory_a, &memory_error) &&
            factory.executor_memory(observed_b->inner(), &memory_b, &memory_error);
        committed_tokens = {memory_a.committed_tokens, memory_b.committed_tokens};
        kv_cache_bytes = {memory_a.kv_cache_bytes, memory_b.kv_cache_bytes};
    }
    const std::size_t released = service.release_completed();
    observed_a.reset();
    observed_b.reset();
    const auto end_to_end_end = Clock::now();
    const auto& metrics = service.metrics();

    const double dispatch_ms = elapsed_ms(dispatch_start, dispatch_end);
    const double prefill_compute_ms = timing_a.prefill_ms + timing_b.prefill_ms;
    const double decode_compute_ms = timing_a.decode_ms + timing_b.decode_ms;
    const double aggregate_compute_ms = prefill_compute_ms + decode_compute_ms;
    const auto contract = factory.decoder_contract();
    const std::size_t kv_bytes_per_token = contract.layer_count *
        contract.layer.kv_head_count * contract.layer.head_dimension * 2 * sizeof(float);
    const bool correctness =
        admitted_a && admitted_b && stats.completed == 2 && stats.generated_tokens == 8 &&
        request_a.generated() == expected_a && request_b.generated() == expected_b &&
        timing_a.prefill_calls == 1 && timing_b.prefill_calls == 1 &&
        timing_a.decode_calls == 3 && timing_b.decode_calls == 3 && cache_isolated &&
        memory_observed &&
        committed_tokens[0] == prompt_a.size() + max_new_tokens - 1 &&
        committed_tokens[1] == prompt_b.size() + max_new_tokens - 1 &&
        kv_cache_bytes[0] == committed_tokens[0] * kv_bytes_per_token &&
        kv_cache_bytes[1] == committed_tokens[1] * kv_bytes_per_token &&
        released == 2 && request_a.cache().empty() && request_b.cache().empty() &&
        weak_a.expired() && weak_b.expired() && metrics.completed_total == 2 &&
        metrics.released_total == 2 && metrics.generated_tokens_total == 8 &&
        metrics.queue_wait_count == 2 && metrics.ttft_count == 2 &&
        metrics.completion_count == 2 &&
        ttft_ms[0] > 0.0 && ttft_ms[1] > 0.0 && completion_ms[0] >= ttft_ms[0] &&
        completion_ms[1] >= ttft_ms[1];

    return {
        prefill_compute_ms,
        decode_compute_ms,
        std::max(0.0, dispatch_ms - aggregate_compute_ms),
        std::max(0.0, aggregate_compute_ms - dispatch_ms),
        dispatch_ms,
        elapsed_ms(end_to_end_start, end_to_end_end),
        dispatch_ms > 0.0 ? 8000.0 / dispatch_ms : 0.0,
        ttft_ms,
        completion_ms,
        committed_tokens,
        kv_cache_bytes,
        correctness,
    };
}

void write_summary(std::ostream& output, const std::string& name, const Summary& summary) {
    output << "    \"" << name << "\": {\"min\": " << summary.minimum
           << ", \"mean\": " << summary.mean << ", \"p50\": " << summary.p50
           << ", \"max\": " << summary.maximum << "}";
}

void write_json(
    const std::string& output_path,
    const std::string& execution_policy,
    const std::string& dispatch_mode,
    std::size_t warmup,
    std::size_t iterations,
    double tokenizer_load_ms,
    double model_load_ms,
    std::size_t model_weight_payload_bytes,
    std::size_t process_peak_rss_bytes,
    const TokenSequence& prompt_a,
    const TokenSequence& prompt_b,
    const std::vector<RunMeasurement>& runs) {
    std::ofstream output(output_path);
    if (!output) throw std::runtime_error("cannot open benchmark result: " + output_path);

    std::vector<double> prefill;
    std::vector<double> decode;
    std::vector<double> overhead;
    std::vector<double> overlap;
    std::vector<double> dispatch;
    std::vector<double> end_to_end;
    std::vector<double> throughput;
    std::vector<double> ttft_a;
    std::vector<double> ttft_b;
    std::vector<double> completion_a;
    std::vector<double> completion_b;
    bool correctness = true;
    for (const auto& run : runs) {
        prefill.push_back(run.prefill_compute_ms);
        decode.push_back(run.decode_compute_ms);
        overhead.push_back(run.runtime_overhead_ms);
        overlap.push_back(run.overlapped_compute_ms);
        dispatch.push_back(run.dispatch_ms);
        end_to_end.push_back(run.end_to_end_ms);
        throughput.push_back(run.throughput_tokens_per_second);
        ttft_a.push_back(run.ttft_ms[0]);
        ttft_b.push_back(run.ttft_ms[1]);
        completion_a.push_back(run.completion_ms[0]);
        completion_b.push_back(run.completion_ms[1]);
        correctness = correctness && run.correctness;
    }

    output << std::setprecision(10)
           << "{\n"
           << "  \"schema_version\": 1,\n"
           << "  \"benchmark\": \"tinyllama_service_core_cpu\",\n"
           << "  \"model\": \"TinyLlama-1.1B-Chat-v1.0.Q4_K_M\",\n"
           << "  \"device\": \"cpu\",\n"
           << "  \"execution_policy\": \"" << execution_policy << "\",\n"
           << "  \"dispatch_mode\": \"" << dispatch_mode << "\",\n"
           << "  \"environment\": {\n"
           << "    \"hardware_concurrency\": " << std::thread::hardware_concurrency() << ",\n"
           << "    \"cpu_model\": \"" << json_escape(cpu_model_name()) << "\",\n"
           << "    \"compiler\": \"" << json_escape(compiler_name()) << "\",\n"
           << "    \"build_type\": \"" << CPP_LLM_ENGINE_BUILD_TYPE << "\"\n"
           << "  },\n"
           << "  \"warmup\": " << warmup << ",\n"
           << "  \"iterations\": " << iterations << ",\n"
           << "  \"request_count\": 2,\n"
           << "  \"max_new_tokens_per_request\": 4,\n"
           << "  \"prompt_tokens\": [" << prompt_a.size() << ", " << prompt_b.size() << "],\n"
           << "  \"one_time_ms\": {\"tokenizer_load\": " << tokenizer_load_ms
           << ", \"model_load\": " << model_load_ms << "},\n"
           << "  \"metrics_ms\": {\n";
    write_summary(output, "prefill_compute", summarize(prefill)); output << ",\n";
    write_summary(output, "decode_compute", summarize(decode)); output << ",\n";
    write_summary(output, "runtime_overhead", summarize(overhead)); output << ",\n";
    write_summary(output, "overlapped_compute", summarize(overlap)); output << ",\n";
    write_summary(output, "dispatch", summarize(dispatch)); output << ",\n";
    write_summary(output, "end_to_end", summarize(end_to_end)); output << ",\n";
    write_summary(output, "request_a_ttft", summarize(ttft_a)); output << ",\n";
    write_summary(output, "request_b_ttft", summarize(ttft_b)); output << ",\n";
    write_summary(output, "request_a_completion", summarize(completion_a)); output << ",\n";
    write_summary(output, "request_b_completion", summarize(completion_b)); output << "\n";
    output << "  },\n"
           << "  \"memory_bytes\": {\n"
           << "    \"model_weight_payload\": " << model_weight_payload_bytes << ",\n"
           << "    \"request_a_kv_committed\": " << runs.front().kv_cache_bytes[0] << ",\n"
           << "    \"request_b_kv_committed\": " << runs.front().kv_cache_bytes[1] << ",\n"
           << "    \"request_a_committed_tokens\": "
           << runs.front().committed_tokens[0] << ",\n"
           << "    \"request_b_committed_tokens\": "
           << runs.front().committed_tokens[1] << ",\n"
           << "    \"kv_committed_total\": "
           << runs.front().kv_cache_bytes[0] + runs.front().kv_cache_bytes[1] << ",\n"
           << "    \"process_peak_rss\": " << process_peak_rss_bytes << "\n"
           << "  },\n"
           << "  \"throughput_tokens_per_second\": ";
    const Summary throughput_summary = summarize(throughput);
    output << "{\"min\": " << throughput_summary.minimum
           << ", \"mean\": " << throughput_summary.mean
           << ", \"p50\": " << throughput_summary.p50
           << ", \"max\": " << throughput_summary.maximum << "},\n"
           << "  \"correctness\": {\n"
           << "    \"all_iterations\": " << (correctness ? "true" : "false") << ",\n"
           << "    \"request_a_golden\": \"29892 306 29915 29885\",\n"
           << "    \"request_b_golden\": \"839 744 29892 263\",\n"
           << "    \"context_and_cache_isolation\": "
           << (correctness ? "true" : "false") << ",\n"
           << "    \"terminal_resources_released\": "
           << (correctness ? "true" : "false") << "\n"
           << "  }\n"
           << "}\n";
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc < 2) {
            std::cerr << "usage: real_tinyllama_serving_benchmark <model.gguf> "
                         "[--warmup N] [--iterations N] [--output PATH] "
                         "[--execution-policy scalar|optimized-avx2] "
                         "[--dispatch-mode sequential|parallel-requests]\n";
            return 2;
        }
        const std::string model_path = argv[1];
        std::size_t warmup = 1;
        std::size_t iterations = 3;
        std::string output_path = "tinyllama_service_cpu_baseline.json";
        std::string execution_policy = "scalar";
        std::string dispatch_mode = "sequential";
        for (int i = 2; i < argc; ++i) {
            const std::string argument = argv[i];
            if (argument == "--warmup" && i + 1 < argc) {
                warmup = static_cast<std::size_t>(std::stoul(argv[++i]));
            } else if (argument == "--iterations" && i + 1 < argc) {
                iterations = static_cast<std::size_t>(std::stoul(argv[++i]));
            } else if (argument == "--output" && i + 1 < argc) {
                output_path = argv[++i];
            } else if (argument == "--execution-policy" && i + 1 < argc) {
                execution_policy = argv[++i];
            } else if (argument == "--dispatch-mode" && i + 1 < argc) {
                dispatch_mode = argv[++i];
            } else {
                std::cerr << "invalid benchmark argument: " << argument << '\n';
                return 2;
            }
        }
        if (iterations == 0) {
            std::cerr << "iterations must be greater than zero\n";
            return 2;
        }
        if (execution_policy != "scalar" && execution_policy != "optimized-avx2") {
            std::cerr << "unsupported execution policy: " << execution_policy << '\n';
            return 2;
        }
        if (dispatch_mode != "sequential" && dispatch_mode != "parallel-requests") {
            std::cerr << "unsupported dispatch mode: " << dispatch_mode << '\n';
            return 2;
        }
        const bool parallel_requests = dispatch_mode == "parallel-requests";

        std::string error;
        const auto tokenizer_start = Clock::now();
        auto tokenizer = cpp_llm_engine::RealTokenizerBackend::load(model_path, &error);
        const double tokenizer_load_ms = elapsed_ms(tokenizer_start, Clock::now());
        const auto model_start = Clock::now();
        const auto execution_mode = execution_policy == "optimized-avx2"
            ? cpp_llm_engine::RealLlamaExecutionMode::OptimizedAvx2
            : cpp_llm_engine::RealLlamaExecutionMode::Scalar;
        auto factory = cpp_llm_engine::RealLlamaExecutorFactory::load(
            model_path, execution_mode, &error);
        const double model_load_ms = elapsed_ms(model_start, Clock::now());
        if (tokenizer == nullptr || factory == nullptr) {
            std::cerr << "benchmark load failed: " << error << '\n';
            return 1;
        }
        const TokenSequence prompt_a = tokenizer->encode("Hello TinyLlama");
        const TokenSequence prompt_b = tokenizer->encode("Hi");

        for (std::size_t i = 0; i < warmup; ++i) {
            if (!run_once(prompt_a, prompt_b, *factory, parallel_requests).correctness) {
                std::cerr << "warmup correctness failed\n";
                return 1;
            }
        }
        std::vector<RunMeasurement> runs;
        runs.reserve(iterations);
        bool correctness = true;
        for (std::size_t i = 0; i < iterations; ++i) {
            runs.push_back(run_once(prompt_a, prompt_b, *factory, parallel_requests));
            correctness = correctness && runs.back().correctness;
        }
        write_json(output_path, execution_policy, dispatch_mode, warmup, iterations,
                   tokenizer_load_ms, model_load_ms,
                   factory->model_weight_payload_bytes(), peak_rss_bytes(),
                   prompt_a, prompt_b, runs);

        std::vector<double> dispatch;
        std::vector<double> throughput;
        for (const auto& run : runs) {
            dispatch.push_back(run.dispatch_ms);
            throughput.push_back(run.throughput_tokens_per_second);
        }
        std::cout << std::setprecision(6)
                  << "benchmark=tinyllama_service_core_cpu\n"
                  << "correctness=" << (correctness ? "PASS" : "FAIL") << '\n'
                  << "execution_policy=" << execution_policy << '\n'
                  << "dispatch_mode=" << dispatch_mode << '\n'
                  << "warmup=" << warmup << '\n'
                  << "iterations=" << iterations << '\n'
                  << "model_load_ms=" << model_load_ms << '\n'
                  << "model_weight_payload_bytes="
                  << factory->model_weight_payload_bytes() << '\n'
                  << "process_peak_rss_bytes=" << peak_rss_bytes() << '\n'
                  << "dispatch_mean_ms=" << summarize(dispatch).mean << '\n'
                  << "throughput_mean_tokens_per_second=" << summarize(throughput).mean << '\n'
                  << "result_file=" << output_path << '\n';
        return correctness ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "benchmark_error=" << error.what() << '\n';
        return 1;
    }
}
