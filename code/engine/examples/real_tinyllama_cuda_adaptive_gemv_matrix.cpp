// Engine CUDA：测量 Adaptive GEMV 在 batch、上下文和 Decode 步数矩阵中的表现。
// Workload：模型只加载一次；每个单元交错运行固定 4 行与自适应路径。
// 约束：Prefill 不计时，逐请求核对顺序执行与 Tensor Batch 的最终 logits。
#include "cpp_llm_engine/real_llama_cuda_executor_factory.h"

#include "chapter06_generation/sampler.h"
#include "chapter07_kv_cache/block_table.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;
using Executor = chapter06_generation::RequestModelExecutor;
using ExecutorPtr = std::shared_ptr<Executor>;
using Logits = chapter06_generation::Logits;

struct Options final {
    std::vector<std::size_t> batches{1, 2, 4, 8};
    std::vector<std::size_t> context_tokens{16, 128};
    std::vector<std::size_t> decode_steps{1, 8};
    std::size_t warmup = 1;
    std::size_t iterations = 3;
    bool adaptive_first = false;
};

struct Sample final {
    double sequential_ms = 0.0;
    double batch_ms = 0.0;
    float max_error = 0.0F;
};

bool parse_list(const std::string& text, std::vector<std::size_t>* output) {
    output->clear();
    std::istringstream stream(text);
    std::string item;
    while (std::getline(stream, item, ',')) {
        char* end = nullptr;
        const unsigned long long value = std::strtoull(item.c_str(), &end, 10);
        if (item.empty() || end == nullptr || *end != '\0' || value == 0 ||
            value > std::numeric_limits<std::size_t>::max()) {
            return false;
        }
        output->push_back(static_cast<std::size_t>(value));
    }
    return !output->empty();
}

bool parse_size(const char* text, std::size_t* output) {
    std::vector<std::size_t> values;
    if (!parse_list(text, &values) || values.size() != 1) return false;
    *output = values.front();
    return true;
}

bool parse_options(int argc, char** argv, Options* options) {
    for (int index = 2; index < argc; index += 2) {
        if (index + 1 >= argc) return false;
        const std::string key = argv[index];
        if (key == "--batches") {
            if (!parse_list(argv[index + 1], &options->batches)) return false;
        } else if (key == "--contexts") {
            if (!parse_list(argv[index + 1], &options->context_tokens)) return false;
        } else if (key == "--decode-steps") {
            if (!parse_list(argv[index + 1], &options->decode_steps)) return false;
        } else if (key == "--warmup") {
            if (!parse_size(argv[index + 1], &options->warmup)) return false;
        } else if (key == "--iterations") {
            if (!parse_size(argv[index + 1], &options->iterations)) return false;
        } else if (key == "--path-order") {
            const std::string value = argv[index + 1];
            if (value != "fixed-first" && value != "adaptive-first") return false;
            options->adaptive_first = value == "adaptive-first";
        } else {
            return false;
        }
    }
    return true;
}

double elapsed_ms(Clock::time_point start, Clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    const std::size_t middle = values.size() / 2;
    return values.size() % 2 == 0
        ? (values[middle - 1] + values[middle]) / 2.0
        : values[middle];
}

float logits_error(const Logits& actual, const Logits& expected) {
    if (actual.size() != expected.size()) return INFINITY;
    float result = 0.0F;
    for (std::size_t index = 0; index < actual.size(); ++index) {
        result = std::max(result, std::fabs(actual[index] - expected[index]));
    }
    return result;
}

bool prepare(
    cpp_llm_engine::RealLlamaCudaExecutorFactory* factory,
    std::size_t request_id_base,
    std::size_t batch_size,
    std::size_t context_tokens,
    std::size_t reserved_tokens,
    std::vector<ExecutorPtr>* executors,
    std::vector<std::int32_t>* next_tokens,
    std::string* error) {
    executors->clear();
    next_tokens->clear();
    chapter06_generation::TokenSequence prompt(context_tokens, 1);
    prompt.back() = 15043;
    chapter06_generation::GreedySampler sampler;
    const auto paged_resource = factory->paged_kv_resource();
    for (std::size_t index = 0; index < batch_size; ++index) {
        auto executor = factory->create_executor();
        Logits logits;
        if (executor == nullptr ||
            !factory->bind_request(executor, request_id_base + index, error) ||
            !executor->prefill(prompt, &logits, error) ||
            paged_resource == nullptr ||
            !paged_resource->reserve(request_id_base + index, reserved_tokens)) {
            return false;
        }
        executors->push_back(std::move(executor));
        next_tokens->push_back(sampler.sample(logits));
    }
    return true;
}

bool run_sample(
    cpp_llm_engine::RealLlamaCudaExecutorFactory* factory,
    std::size_t request_id_base,
    std::size_t batch_size,
    std::size_t context_tokens,
    std::size_t decode_steps,
    bool batch_first,
    Sample* sample,
    std::string* error) {
    std::vector<ExecutorPtr> sequential;
    std::vector<ExecutorPtr> batch;
    std::vector<std::int32_t> sequential_tokens;
    std::vector<std::int32_t> batch_tokens;
    if (!prepare(factory, request_id_base, batch_size, context_tokens,
                 context_tokens + decode_steps,
                 &sequential, &sequential_tokens, error) ||
        !prepare(factory, request_id_base + batch_size, batch_size, context_tokens,
                 context_tokens + decode_steps,
                 &batch, &batch_tokens, error) ||
        sequential_tokens != batch_tokens) {
        return false;
    }

    chapter06_generation::GreedySampler sampler;
    std::vector<Logits> sequential_logits(batch_size);
    std::vector<Logits> batch_logits;
    const auto run_sequential = [&] {
        const auto start = Clock::now();
        for (std::size_t step = 0; step < decode_steps; ++step) {
            for (std::size_t index = 0; index < batch_size; ++index) {
                if (!sequential[index]->decode(
                        sequential_tokens[index], &sequential_logits[index], error)) {
                    return false;
                }
                sequential_tokens[index] = sampler.sample(sequential_logits[index]);
            }
        }
        sample->sequential_ms = elapsed_ms(start, Clock::now());
        return true;
    };
    const auto run_batch = [&] {
        const auto start = Clock::now();
        const auto paged_resource = factory->paged_kv_resource();
        if (paged_resource == nullptr) {
            if (error != nullptr) *error = "matrix paged KV resource is unavailable";
            return false;
        }
        for (std::size_t step = 0; step < decode_steps; ++step) {
            std::vector<std::size_t> positions(
                batch_size, context_tokens + step);
            std::vector<std::size_t> request_ids(batch_size);
            for (std::size_t index = 0; index < batch_size; ++index) {
                request_ids[index] = request_id_base + batch_size + index;
            }
            if (!factory->decode_batch(
                    batch, batch_tokens, positions, request_ids,
                    &batch_logits, error)) {
                return false;
            }
            for (std::size_t index = 0; index < batch_size; ++index) {
                if (!paged_resource->commit_tokens(
                        request_ids[index], context_tokens + step + 1)) {
                    if (error != nullptr) {
                        *error = "matrix cannot commit paged KV tokens after batch Decode";
                    }
                    return false;
                }
                batch_tokens[index] = sampler.sample(batch_logits[index]);
            }
        }
        sample->batch_ms = elapsed_ms(start, Clock::now());
        return true;
    };
    const bool succeeded = batch_first
        ? run_batch() && run_sequential()
        : run_sequential() && run_batch();
    if (!succeeded || batch_logits.size() != batch_size) return false;

    for (std::size_t index = 0; index < batch_size; ++index) {
        sample->max_error = std::max(
            sample->max_error,
            logits_error(batch_logits[index], sequential_logits[index]));
        if (sampler.sample(batch_logits[index]) !=
            sampler.sample(sequential_logits[index])) {
            sample->max_error = INFINITY;
        }
    }
    for (std::size_t index = 0; index < batch_size * 2; ++index) {
        factory->release_request(request_id_base + index);
    }
    return sample->max_error <= 1.0e-3F;
}

const char* path_name(cpp_llm_engine::CudaQuantizedGemvMode mode) {
    return mode == cpp_llm_engine::CudaQuantizedGemvMode::Adaptive
        ? "adaptive" : "fixed-four";
}

bool run_cell(
    cpp_llm_engine::RealLlamaCudaExecutorFactory* factory,
    cpp_llm_engine::CudaQuantizedGemvMode mode,
    std::size_t batch_size,
    std::size_t context_tokens,
    std::size_t decode_steps,
    const Options& options,
    std::size_t* request_id,
    std::string* error) {
    factory->set_quantized_gemv_mode(mode);
    std::cerr << "matrix_cell path=" << path_name(mode)
              << " batch=" << batch_size
              << " context=" << context_tokens
              << " decode_steps=" << decode_steps << '\n';
    std::vector<double> sequential_samples;
    std::vector<double> batch_samples;
    float maximum_error = 0.0F;
    for (std::size_t index = 0;
         index < options.warmup + options.iterations; ++index) {
        Sample sample;
        if (!run_sample(
                factory, *request_id, batch_size, context_tokens, decode_steps,
                index % 2 != 0, &sample, error)) {
            return false;
        }
        *request_id += batch_size * 2;
        maximum_error = std::max(maximum_error, sample.max_error);
        if (index >= options.warmup) {
            sequential_samples.push_back(sample.sequential_ms);
            batch_samples.push_back(sample.batch_ms);
        }
    }
    const double sequential = median(sequential_samples);
    const double batched = median(batch_samples);
    std::cout << path_name(mode) << ',' << batch_size << ',' << context_tokens
              << ',' << decode_steps << ',' << sequential << ',' << batched
              << ',' << batched / sequential << ',' << maximum_error << '\n'
              << std::flush;
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    Options options;
    if (argc < 2 || !parse_options(argc, argv, &options)) {
        std::cerr << "usage: real_tinyllama_cuda_adaptive_gemv_matrix <model.gguf> "
                     "[--batches 1,2,4,8] [--contexts 16,128] "
                     "[--decode-steps 1,8] [--warmup 1] [--iterations 3]\n";
        return 2;
    }
    const std::size_t maximum_batch =
        *std::max_element(options.batches.begin(), options.batches.end());
    const std::size_t maximum_context =
        *std::max_element(options.context_tokens.begin(), options.context_tokens.end());
    const std::size_t maximum_steps =
        *std::max_element(options.decode_steps.begin(), options.decode_steps.end());
    const std::size_t capacity = maximum_context + maximum_steps;
    constexpr std::size_t block_size = 8;
    const std::size_t blocks_per_request =
        (capacity + block_size - 1) / block_size;

    std::string error;
    auto factory = cpp_llm_engine::RealLlamaCudaExecutorFactory::load(
        argv[1], capacity, &error);
    if (factory == nullptr || !factory->configure_paged_kv_cache(
            maximum_batch * 2 * blocks_per_request + 8,
            block_size,
            &error)) {
        std::cerr << error << '\n';
        return 1;
    }

    std::cout << "path,batch,context_tokens,decode_steps,sequential_median_ms,"
                 "tensor_batch_median_ms,batch_over_sequential,logits_max_error\n"
              << std::flush;
    std::size_t request_id = 1000;
    std::size_t cell = 0;
    for (const std::size_t batch_size : options.batches) {
        for (const std::size_t context_tokens : options.context_tokens) {
            for (const std::size_t decode_steps : options.decode_steps) {
                const auto first = (cell++ + (options.adaptive_first ? 1 : 0)) % 2 == 0
                    ? cpp_llm_engine::CudaQuantizedGemvMode::FixedTile4
                    : cpp_llm_engine::CudaQuantizedGemvMode::Adaptive;
                const auto second = first == cpp_llm_engine::CudaQuantizedGemvMode::Adaptive
                    ? cpp_llm_engine::CudaQuantizedGemvMode::FixedTile4
                    : cpp_llm_engine::CudaQuantizedGemvMode::Adaptive;
                if (!run_cell(factory.get(), first, batch_size, context_tokens,
                              decode_steps, options, &request_id, &error) ||
                    !run_cell(factory.get(), second, batch_size, context_tokens,
                              decode_steps, options, &request_id, &error)) {
                    std::cerr << "adaptive GEMV matrix failed: " << error << '\n';
                    return 1;
                }
            }
        }
    }
    return 0;
}
