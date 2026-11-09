// Engine CUDA：比较分页 K/V 下三阶段与融合在线 Attention 的真实墙钟时间。
// Workload：32-token Tensor Prefill 加一次 Decode；每组使用新请求并交替执行顺序。
// 约束：先核对 logits 和 greedy token；Context 创建与绑定不计时。
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
using Logits = chapter06_generation::Logits;

struct Sample final {
    double prefill_ms = 0.0;
    double decode_ms = 0.0;
    Logits prefill_logits;
    Logits decode_logits;
};

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

bool run_sample(
    cpp_llm_engine::RealLlamaCudaExecutorFactory* factory,
    cpp_llm_engine::CudaAttentionMode mode,
    const chapter06_generation::TokenSequence& prompt,
    std::size_t request_id,
    Sample* sample,
    std::string* error) {
    factory->set_attention_mode(mode);
    auto executor = factory->create_executor();
    if (executor == nullptr || sample == nullptr ||
        !factory->bind_request(executor, request_id, error)) {
        return false;
    }
    const auto prefill_start = Clock::now();
    const bool prefill_ok = executor->prefill(prompt, &sample->prefill_logits, error);
    const auto prefill_end = Clock::now();
    if (!prefill_ok) return false;
    chapter06_generation::GreedySampler sampler;
    const auto decode_start = Clock::now();
    const bool decode_ok = executor->decode(
        sampler.sample(sample->prefill_logits), &sample->decode_logits, error);
    const auto decode_end = Clock::now();
    sample->prefill_ms = std::chrono::duration<double, std::milli>(
        prefill_end - prefill_start).count();
    sample->decode_ms = std::chrono::duration<double, std::milli>(
        decode_end - decode_start).count();
    factory->release_request(request_id);
    return decode_ok;
}

void print_samples(const char* name, const std::vector<double>& samples) {
    std::cout << name << ':';
    for (const double sample : samples) std::cout << ' ' << sample;
    std::cout << '\n';
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "usage: real_tinyllama_cuda_paged_attention_benchmark <model.gguf>\n";
        return 2;
    }
    std::string error;
    auto factory = cpp_llm_engine::RealLlamaCudaExecutorFactory::load(
        argv[1], 64, &error);
    if (factory == nullptr || !factory->configure_paged_kv_cache(128, 8, &error)) {
        std::cerr << error << '\n';
        return 1;
    }
    chapter06_generation::TokenSequence prompt(32);
    for (std::size_t index = 0; index < prompt.size(); ++index) {
        prompt[index] = index % 2 == 0 ? 1 : 15043;
    }

    constexpr std::size_t warmup = 2;
    constexpr std::size_t iterations = 10;
    std::vector<double> baseline_prefill;
    std::vector<double> fused_prefill;
    std::vector<double> baseline_decode;
    std::vector<double> fused_decode;
    float prefill_error = 0.0F;
    float decode_error = 0.0F;
    chapter06_generation::GreedySampler sampler;
    for (std::size_t index = 0; index < warmup + iterations; ++index) {
        Sample baseline;
        Sample fused;
        const std::size_t request_id = 3000 + index * 2;
        const bool succeeded = index % 2 == 0
            ? run_sample(factory.get(), cpp_llm_engine::CudaAttentionMode::ThreeStage,
                         prompt, request_id, &baseline, &error) &&
                run_sample(factory.get(), cpp_llm_engine::CudaAttentionMode::FusedOnline,
                           prompt, request_id + 1, &fused, &error)
            : run_sample(factory.get(), cpp_llm_engine::CudaAttentionMode::FusedOnline,
                         prompt, request_id + 1, &fused, &error) &&
                run_sample(factory.get(), cpp_llm_engine::CudaAttentionMode::ThreeStage,
                           prompt, request_id, &baseline, &error);
        if (!succeeded) {
            std::cerr << error << '\n';
            return 1;
        }
        prefill_error = std::max(
            prefill_error, max_error(fused.prefill_logits, baseline.prefill_logits));
        decode_error = std::max(
            decode_error, max_error(fused.decode_logits, baseline.decode_logits));
        if (prefill_error > 1.0e-3F || decode_error > 1.0e-3F ||
            sampler.sample(fused.prefill_logits) !=
                sampler.sample(baseline.prefill_logits) ||
            sampler.sample(fused.decode_logits) !=
                sampler.sample(baseline.decode_logits)) {
            std::cerr << "paged attention A/B correctness gate failed\n";
            return 1;
        }
        if (index >= warmup) {
            baseline_prefill.push_back(baseline.prefill_ms);
            fused_prefill.push_back(fused.prefill_ms);
            baseline_decode.push_back(baseline.decode_ms);
            fused_decode.push_back(fused.decode_ms);
        }
    }
    const double baseline_prefill_median = median(baseline_prefill);
    const double fused_prefill_median = median(fused_prefill);
    const double baseline_decode_median = median(baseline_decode);
    const double fused_decode_median = median(fused_decode);
    std::cout << "prompt_tokens: 32\npaged_kv: true\nblock_size: 8\n"
              << "warmup: 2\niterations: 10\nstatistic: median\n"
              << "three_stage_prefill_median_ms: " << baseline_prefill_median << '\n'
              << "fused_online_prefill_median_ms: " << fused_prefill_median << '\n'
              << "prefill_fused_over_three_stage: "
              << fused_prefill_median / baseline_prefill_median << '\n'
              << "three_stage_decode_median_ms: " << baseline_decode_median << '\n'
              << "fused_online_decode_median_ms: " << fused_decode_median << '\n'
              << "decode_fused_over_three_stage: "
              << fused_decode_median / baseline_decode_median << '\n'
              << "prefill_max_error: " << prefill_error << '\n'
              << "decode_max_error: " << decode_error << '\n';
    print_samples("three_stage_prefill_samples_ms", baseline_prefill);
    print_samples("fused_online_prefill_samples_ms", fused_prefill);
    print_samples("three_stage_decode_samples_ms", baseline_decode);
    print_samples("fused_online_decode_samples_ms", fused_decode);
    return 0;
}
