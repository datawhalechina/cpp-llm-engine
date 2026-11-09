// Engine CUDA：比较分页 K/V 的逐 token Prefill 与 Tensor Prefill 墙钟时间。
// Workload：32-token prompt、2 次预热、10 次测量；Context 创建和绑定不计时。
// 约束：先核对 logits 和 greedy token，再解释时间；本程序不注册为 CTest。
#include "cpp_llm_engine/real_llama_cuda_executor_factory.h"

#include "chapter06_generation/sampler.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <string>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

struct Result final {
    double median_ms = 0.0;
    std::vector<double> samples_ms;
    chapter06_generation::Logits logits;
};

double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    return values[values.size() / 2];
}

bool run_iteration(
    cpp_llm_engine::RealLlamaCudaExecutorFactory* factory,
    const chapter06_generation::TokenSequence& prompt,
    std::size_t request_id,
    bool tensor_prefill,
    chapter06_generation::Logits* logits,
    double* elapsed_ms,
    std::string* error) {
    auto executor = factory->create_executor();
    if (executor == nullptr || logits == nullptr || elapsed_ms == nullptr ||
        !factory->bind_request(executor, request_id, error)) {
        return false;
    }
    const auto start = Clock::now();
    bool succeeded = false;
    if (tensor_prefill) {
        succeeded = executor->prefill(prompt, logits, error);
    } else {
        for (const auto token : prompt) {
            if (!executor->decode(token, logits, error)) break;
        }
        succeeded = executor->execution_state().committed_tokens == prompt.size();
    }
    const auto end = Clock::now();
    *elapsed_ms =
        std::chrono::duration<double, std::milli>(end - start).count();
    factory->release_request(request_id);
    return succeeded;
}

bool run_mode(
    cpp_llm_engine::RealLlamaCudaExecutorFactory* factory,
    const chapter06_generation::TokenSequence& prompt,
    bool tensor_prefill,
    std::size_t request_id_base,
    Result* result,
    std::string* error) {
    constexpr std::size_t warmup = 2;
    constexpr std::size_t iterations = 10;
    if (factory == nullptr || result == nullptr) return false;
    for (std::size_t index = 0; index < warmup; ++index) {
        double ignored = 0.0;
        if (!run_iteration(
                factory, prompt, request_id_base + index, tensor_prefill,
                &result->logits, &ignored, error)) {
            return false;
        }
    }
    result->samples_ms.clear();
    result->samples_ms.reserve(iterations);
    for (std::size_t index = 0; index < iterations; ++index) {
        double elapsed = 0.0;
        if (!run_iteration(
                factory, prompt, request_id_base + warmup + index,
                tensor_prefill, &result->logits, &elapsed, error)) {
            return false;
        }
        result->samples_ms.push_back(elapsed);
    }
    result->median_ms = median(result->samples_ms);
    return true;
}

float max_error(
    const chapter06_generation::Logits& actual,
    const chapter06_generation::Logits& expected) {
    if (actual.size() != expected.size()) return INFINITY;
    float error = 0.0F;
    for (std::size_t index = 0; index < actual.size(); ++index) {
        error = std::max(error, std::fabs(actual[index] - expected[index]));
    }
    return error;
}

void print_samples(const char* name, const std::vector<double>& samples) {
    std::cout << name << ':';
    for (const double sample : samples) std::cout << ' ' << sample;
    std::cout << '\n';
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "usage: real_tinyllama_cuda_paged_prefill_benchmark <model.gguf>\n";
        return 2;
    }
    std::string error;
    auto factory = cpp_llm_engine::RealLlamaCudaExecutorFactory::load(
        argv[1], 64, &error);
    if (factory == nullptr || !factory->configure_paged_kv_cache(8, 8, &error)) {
        std::cerr << error << '\n';
        return 1;
    }
    chapter06_generation::TokenSequence prompt(32);
    for (std::size_t index = 0; index < prompt.size(); ++index) {
        prompt[index] = index % 2 == 0 ? 1 : 15043;
    }
    Result sequential;
    Result tensor;
    if (!run_mode(factory.get(), prompt, false, 1000, &sequential, &error) ||
        !run_mode(factory.get(), prompt, true, 2000, &tensor, &error)) {
        std::cerr << error << '\n';
        return 1;
    }
    chapter06_generation::GreedySampler sampler;
    const float error_value = max_error(tensor.logits, sequential.logits);
    if (error_value > 1.0e-3F ||
        sampler.sample(tensor.logits) != sampler.sample(sequential.logits)) {
        std::cerr << "paged Prefill A/B correctness gate failed\n";
        return 1;
    }
    std::cout << "prompt_tokens: 32\nwarmup: 2\niterations: 10\n"
              << "statistic: median\n"
              << "sequential_prefill_median_ms: " << sequential.median_ms << '\n'
              << "tensor_prefill_median_ms: " << tensor.median_ms << '\n'
              << "tensor_over_sequential: "
              << tensor.median_ms / sequential.median_ms << '\n'
              << "logits_max_error: " << error_value << '\n';
    print_samples("sequential_prefill_samples_ms", sequential.samples_ms);
    print_samples("tensor_prefill_samples_ms", tensor.samples_ms);
    return 0;
}
