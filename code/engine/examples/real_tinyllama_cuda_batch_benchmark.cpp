// Engine CUDA：比较双请求分页顺序 Decode 与 Tensor Batch 的墙钟时间。
// Workload：两个不同长度 prompt、2 次预热、10 组配对测量；Prefill 不计时。
// 约束：逐请求核对 logits 与 greedy token；本程序不注册为 CTest。
#include "cpp_llm_engine/real_llama_cuda_executor_factory.h"

#include "chapter06_generation/sampler.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;
using Executor = chapter06_generation::RequestModelExecutor;
using ExecutorPtr = std::shared_ptr<Executor>;
using Logits = chapter06_generation::Logits;

struct PairResult final {
    double sequential_ms = 0.0;
    double batch_ms = 0.0;
    float max_error = 0.0F;
};

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

float max_error(const Logits& actual, const Logits& expected) {
    if (actual.size() != expected.size()) return INFINITY;
    float result = 0.0F;
    for (std::size_t index = 0; index < actual.size(); ++index) {
        result = std::max(result, std::fabs(actual[index] - expected[index]));
    }
    return result;
}

bool prepare(
    cpp_llm_engine::RealLlamaCudaExecutorFactory* factory,
    std::size_t first_request_id,
    ExecutorPtr* first,
    ExecutorPtr* second,
    std::vector<std::int32_t>* next_tokens,
    std::string* error) {
    *first = factory->create_executor();
    *second = factory->create_executor();
    if (*first == nullptr || *second == nullptr ||
        !factory->bind_request(*first, first_request_id, error) ||
        !factory->bind_request(*second, first_request_id + 1, error)) {
        return false;
    }
    Logits first_logits;
    Logits second_logits;
    if (!(*first)->prefill({15043}, &first_logits, error) ||
        !(*second)->prefill({1, 15043}, &second_logits, error)) {
        return false;
    }
    chapter06_generation::GreedySampler sampler;
    *next_tokens = {sampler.sample(first_logits), sampler.sample(second_logits)};
    return true;
}

bool run_pair(
    cpp_llm_engine::RealLlamaCudaExecutorFactory* factory,
    std::size_t request_id_base,
    bool batch_first,
    PairResult* result,
    std::string* error) {
    ExecutorPtr sequential_a;
    ExecutorPtr sequential_b;
    ExecutorPtr batch_a;
    ExecutorPtr batch_b;
    std::vector<std::int32_t> sequential_tokens;
    std::vector<std::int32_t> batch_tokens;
    if (!prepare(factory, request_id_base, &sequential_a, &sequential_b,
                 &sequential_tokens, error) ||
        !prepare(factory, request_id_base + 2, &batch_a, &batch_b,
                 &batch_tokens, error) ||
        sequential_tokens != batch_tokens) {
        return false;
    }

    Logits sequential_a_logits;
    Logits sequential_b_logits;
    std::vector<Logits> batch_logits;
    const auto run_sequential = [&] {
        const auto start = Clock::now();
        const bool succeeded = sequential_a->decode(
            sequential_tokens[0], &sequential_a_logits, error) &&
            sequential_b->decode(
                sequential_tokens[1], &sequential_b_logits, error);
        result->sequential_ms = elapsed_ms(start, Clock::now());
        return succeeded;
    };
    const auto run_batch = [&] {
        const auto start = Clock::now();
        const bool succeeded = factory->decode_batch(
            {batch_a, batch_b}, batch_tokens, {1, 2},
            {request_id_base + 2, request_id_base + 3},
            &batch_logits, error);
        result->batch_ms = elapsed_ms(start, Clock::now());
        return succeeded;
    };
    const bool succeeded = batch_first
        ? run_batch() && run_sequential()
        : run_sequential() && run_batch();
    if (!succeeded || batch_logits.size() != 2) return false;

    result->max_error = std::max(
        max_error(batch_logits[0], sequential_a_logits),
        max_error(batch_logits[1], sequential_b_logits));
    chapter06_generation::GreedySampler sampler;
    const bool correct = result->max_error <= 1.0e-3F &&
        sampler.sample(batch_logits[0]) == sampler.sample(sequential_a_logits) &&
        sampler.sample(batch_logits[1]) == sampler.sample(sequential_b_logits);
    for (std::size_t offset = 0; offset < 4; ++offset) {
        factory->release_request(request_id_base + offset);
    }
    return correct;
}

void print_samples(const char* name, const std::vector<double>& samples) {
    std::cout << name << ':';
    for (const double sample : samples) std::cout << ' ' << sample;
    std::cout << '\n';
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 4 || std::string(argv[2]) != "--gemv-path") {
        std::cerr << "usage: real_tinyllama_cuda_batch_benchmark <model.gguf> "
                     "--gemv-path fixed-four|adaptive\n";
        return 2;
    }
    const std::string gemv_path = argv[3];
    if (gemv_path != "fixed-four" && gemv_path != "adaptive") {
        std::cerr << "unsupported GEMV path: " << gemv_path << '\n';
        return 2;
    }
    std::string error;
    auto factory = cpp_llm_engine::RealLlamaCudaExecutorFactory::load(
        argv[1], 16, &error);
    if (factory == nullptr || !factory->configure_paged_kv_cache(64, 8, &error)) {
        std::cerr << error << '\n';
        return 1;
    }
    factory->set_quantized_gemv_mode(
        gemv_path == "adaptive"
            ? cpp_llm_engine::CudaQuantizedGemvMode::Adaptive
            : cpp_llm_engine::CudaQuantizedGemvMode::FixedTile4);

    constexpr std::size_t warmup = 2;
    constexpr std::size_t iterations = 10;
    std::vector<double> sequential_samples;
    std::vector<double> batch_samples;
    float observed_max_error = 0.0F;
    for (std::size_t index = 0; index < warmup + iterations; ++index) {
        PairResult result;
        if (!run_pair(
                factory.get(), 1000 + index * 4, index % 2 != 0,
                &result, &error)) {
            std::cerr << "CUDA batch A/B failed: " << error << '\n';
            return 1;
        }
        observed_max_error = std::max(observed_max_error, result.max_error);
        if (index >= warmup) {
            sequential_samples.push_back(result.sequential_ms);
            batch_samples.push_back(result.batch_ms);
        }
    }
    const double sequential_median = median(sequential_samples);
    const double batch_median = median(batch_samples);
    std::cout << "gemv_path: " << gemv_path
              << "\nrequests: 2\nprompt_tokens: 1,2\ndecode_steps: 1\n"
              << "paged_kv: true\nblock_size: 8\nwarmup: " << warmup
              << "\niterations: " << iterations << "\nstatistic: median\n"
              << "sequential_decode_median_ms: " << sequential_median << '\n'
              << "tensor_batch_decode_median_ms: " << batch_median << '\n'
              << "batch_over_sequential: " << batch_median / sequential_median << '\n'
              << "logits_max_error: " << observed_max_error << '\n';
    print_samples("sequential_decode_samples_ms", sequential_samples);
    print_samples("tensor_batch_decode_samples_ms", batch_samples);
    return 0;
}
