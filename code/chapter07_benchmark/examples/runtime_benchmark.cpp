// Chapter 07：测量 Chapter 06 CPU Runtime 的单请求 prefill/decode 基线。
// 职责：固定教学 Transformer、输入和迭代参数，输出正确性与阶段耗时。
// 验证：结果必须包含 correctness=PASS，性能数字仅用于同环境比较。
#include "chapter06_generation/embedding.h"
#include "chapter06_generation/position_encoding.h"
#include "chapter06_generation/request.h"
#include "chapter06_generation/runtime.h"
#include "chapter06_generation/sampler.h"
#include "chapter06_generation/transformer.h"

#include <chrono>
#include <cstddef>
#include <iostream>

namespace {

struct Measurement {
    double prefill_ms = 0.0;
    double decode_ms = 0.0;
    std::size_t generated_tokens = 0;
    bool correct = false;
};

Measurement run_once() {
    using clock = std::chrono::steady_clock;
    chapter06_generation::GenerationConfig config;
    config.max_new_tokens = 4;
    config.max_context_tokens = 32;

    chapter06_generation::Embedding embedding(64);
    chapter06_generation::PositionEncoding position_encoding(32);
    chapter06_generation::Transformer transformer(2, embedding, position_encoding);
    chapter06_generation::Runtime runtime(config, transformer);
    chapter06_generation::Request request({1, 2, 3, 4});
    chapter06_generation::GreedySampler sampler;

    const auto prefill_start = clock::now();
    const std::size_t prefills = runtime.prefill(request);
    const auto prefill_end = clock::now();
    const auto decode_start = clock::now();
    const std::size_t generated = runtime.decode(request, sampler);
    const auto decode_end = clock::now();

    const double prefill_ms =
        std::chrono::duration<double, std::milli>(prefill_end - prefill_start).count();
    const double decode_ms =
        std::chrono::duration<double, std::milli>(decode_end - decode_start).count();
    return {prefill_ms, decode_ms, generated,
            runtime.ready() && prefills == request.prompt_size() &&
                generated == config.max_new_tokens && request.finished() &&
                request.cache().size() == request.prompt_size() + generated};
}

}  // namespace

int main() {
    constexpr std::size_t warmup = 1;
    constexpr std::size_t iterations = 3;
    for (std::size_t i = 0; i < warmup; ++i) {
        (void)run_once();
    }

    Measurement total;
    for (std::size_t i = 0; i < iterations; ++i) {
        const Measurement measurement = run_once();
        total.prefill_ms += measurement.prefill_ms;
        total.decode_ms += measurement.decode_ms;
        total.generated_tokens += measurement.generated_tokens;
        total.correct = total.correct || measurement.correct;
    }

    const double prefill_ms = total.prefill_ms / iterations;
    const double decode_ms = total.decode_ms / iterations;
    const double total_ms = prefill_ms + decode_ms;
    const double tokens_per_second = total_ms > 0.0
                                         ? static_cast<double>(total.generated_tokens) *
                                               1000.0 / (total_ms * iterations)
                                         : 0.0;
    std::cout << "runtime_benchmark=chapter06_cpu\n"
              << "correctness=" << (total.correct ? "PASS" : "FAIL") << '\n'
              << "warmup=" << warmup << '\n'
              << "iterations=" << iterations << '\n'
              << "prefill_ms=" << prefill_ms << '\n'
              << "decode_ms=" << decode_ms << '\n'
              << "total_ms=" << total_ms << '\n'
              << "generated_tokens_per_iteration=" << total.generated_tokens / iterations << '\n'
              << "tokens_per_second=" << tokens_per_second << '\n';
    return total.correct ? 0 : 1;
}
