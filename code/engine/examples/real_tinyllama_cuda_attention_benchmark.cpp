// Engine CUDA：比较三阶段与融合 Attention 的真实 Prompt Prefill/Decode 墙钟时间。
// Workload：32-token prompt、2 次 warmup、15 次测量，同一 Executor 复用 workspace。
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
    double prefill_ms = 0.0;
    double decode_ms = 0.0;
    std::vector<double> prefill_samples;
    std::vector<double> decode_samples;
    chapter06_generation::Logits prefill_logits;
    chapter06_generation::Logits decode_logits;
};

double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    return values[values.size() / 2];
}

bool run_mode(
    cpp_llm_engine::RealLlamaCudaExecutorFactory* factory,
    cpp_llm_engine::CudaAttentionMode mode,
    const chapter06_generation::TokenSequence& prompt,
    Result* result,
    std::string* error) {
    constexpr int warmup = 2;
    constexpr int iterations = 15;
    factory->set_attention_mode(mode);
    auto executor = factory->create_executor();
    if (executor == nullptr || result == nullptr) return false;
    chapter06_generation::GreedySampler sampler;
    for (int iteration = 0; iteration < warmup; ++iteration) {
        if (!executor->prefill(prompt, &result->prefill_logits, error)) return false;
        const auto token = sampler.sample(result->prefill_logits);
        if (!executor->decode(token, &result->decode_logits, error)) return false;
    }
    result->prefill_samples.clear();
    result->decode_samples.clear();
    result->prefill_samples.reserve(iterations);
    result->decode_samples.reserve(iterations);
    for (int iteration = 0; iteration < iterations; ++iteration) {
        const auto prefill_start = Clock::now();
        if (!executor->prefill(prompt, &result->prefill_logits, error)) return false;
        const auto prefill_end = Clock::now();
        const auto token = sampler.sample(result->prefill_logits);
        const auto decode_start = Clock::now();
        if (!executor->decode(token, &result->decode_logits, error)) return false;
        const auto decode_end = Clock::now();
        result->prefill_samples.push_back(
            std::chrono::duration<double, std::milli>(prefill_end - prefill_start).count());
        result->decode_samples.push_back(
            std::chrono::duration<double, std::milli>(decode_end - decode_start).count());
    }
    result->prefill_ms = median(result->prefill_samples);
    result->decode_ms = median(result->decode_samples);
    return true;
}

void print_samples(const char* name, const std::vector<double>& samples) {
    std::cout << name << ':';
    for (const double sample : samples) std::cout << ' ' << sample;
    std::cout << '\n';
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

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "usage: real_tinyllama_cuda_attention_benchmark <model.gguf>\n";
        return 2;
    }
    std::string error;
    auto factory = cpp_llm_engine::RealLlamaCudaExecutorFactory::load(
        argv[1], 64, &error);
    if (factory == nullptr) {
        std::cerr << error << '\n';
        return 1;
    }
    chapter06_generation::TokenSequence prompt(32);
    for (std::size_t index = 0; index < prompt.size(); ++index) {
        prompt[index] = index % 2 == 0 ? 1 : 15043;
    }
    Result baseline;
    Result fused;
    if (!run_mode(
            factory.get(), cpp_llm_engine::CudaAttentionMode::ThreeStage,
            prompt, &baseline, &error) ||
        !run_mode(
            factory.get(), cpp_llm_engine::CudaAttentionMode::FusedOnline,
            prompt, &fused, &error)) {
        std::cerr << error << '\n';
        return 1;
    }
    chapter06_generation::GreedySampler sampler;
    const float prefill_error = max_error(fused.prefill_logits, baseline.prefill_logits);
    const float decode_error = max_error(fused.decode_logits, baseline.decode_logits);
    if (prefill_error > 1.0e-3F || decode_error > 1.0e-3F ||
        sampler.sample(fused.prefill_logits) != sampler.sample(baseline.prefill_logits) ||
        sampler.sample(fused.decode_logits) != sampler.sample(baseline.decode_logits)) {
        std::cerr << "attention A/B correctness gate failed\n";
        return 1;
    }
    std::cout << "prompt_tokens: " << prompt.size() << '\n'
              << "warmup: 2\niterations: 15\nstatistic: median\n"
              << "three_stage_prefill_median_ms: " << baseline.prefill_ms << '\n'
              << "fused_online_prefill_median_ms: " << fused.prefill_ms << '\n'
              << "three_stage_decode_median_ms: " << baseline.decode_ms << '\n'
              << "fused_online_decode_median_ms: " << fused.decode_ms << '\n'
              << "prefill_max_error: " << prefill_error << '\n'
              << "decode_max_error: " << decode_error << '\n';
    print_samples("three_stage_prefill_samples_ms", baseline.prefill_samples);
    print_samples("fused_online_prefill_samples_ms", fused.prefill_samples);
    print_samples("three_stage_decode_samples_ms", baseline.decode_samples);
    print_samples("fused_online_decode_samples_ms", fused.decode_samples);
    return 0;
}
