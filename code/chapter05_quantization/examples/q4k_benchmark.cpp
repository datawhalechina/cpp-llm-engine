// Chapter 05：测量 Q4_K reference、scalar fused 与 AVX2 Linear。
// 约束：结果只用于同一机器、构建和 workload 下的实现对照。
// 性能结论由 Chapter09 结合环境、warmup 和多轮统计登记。
#include "chapter05_quantization/q4k_dequantize.h"
#include "chapter05_quantization/q4k_linear.h"

#include <chrono>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

int main() {
    constexpr std::size_t rows = 256;
    constexpr std::size_t cols = 2048;
    constexpr std::size_t elements = rows * cols;
    constexpr std::size_t bytes = elements / chapter05_quantization::kQ4KBlockElements *
                                   chapter05_quantization::kQ4KBlockBytes;

    std::vector<std::uint8_t> weights(bytes, 0);
    for (std::size_t block = 0; block < bytes / 144; ++block) {
        weights[block * 144 + 1] = 0x3c;
        for (std::size_t i = block * 144 + 4; i < block * 144 + 12; ++i) weights[i] = 1;
        for (std::size_t i = block * 144 + 12; i < block * 144 + 144; ++i) weights[i] = 0x11;
    }
    const std::vector<float> input(cols, 1.0f);
    std::vector<float> decoded;
    std::vector<float> output;
    std::vector<float> fused_output;
    std::vector<float> avx2_output;
    std::string error;

    constexpr int iterations = 5;
    const auto decode_start = std::chrono::steady_clock::now();
    for (int i = 0; i < iterations; ++i) {
        if (!chapter05_quantization::dequantize_q4_k(weights.data(), weights.size(), elements, &decoded, &error)) {
            std::cerr << error << '\n';
            return 1;
        }
    }
    const auto decode_end = std::chrono::steady_clock::now();

    const auto linear_start = std::chrono::steady_clock::now();
    for (int i = 0; i < iterations; ++i) {
        if (!chapter05_quantization::Q4KLinearReference::forward(
                weights.data(), weights.size(), rows, cols, input, &output, &error)) {
            std::cerr << error << '\n';
            return 1;
        }
    }
    const auto linear_end = std::chrono::steady_clock::now();

    const auto fused_start = std::chrono::steady_clock::now();
    for (int i = 0; i < iterations; ++i) {
        if (!chapter05_quantization::Q4KLinearFused::forward(
                weights.data(), weights.size(), rows, cols, input, &fused_output, &error)) {
            std::cerr << error << '\n';
            return 1;
        }
    }
    const auto fused_end = std::chrono::steady_clock::now();

#ifdef CHAPTER05_HAS_AVX2
    double avx2_ms = 0.0;
    const bool has_avx2 = chapter05_quantization::q4k_avx2_supported();
    if (has_avx2) {
        const auto avx2_start = std::chrono::steady_clock::now();
        for (int i = 0; i < iterations; ++i) {
            if (!chapter05_quantization::Q4KLinearAvx2::forward(
                    weights.data(), weights.size(), rows, cols, input,
                    &avx2_output, &error)) {
                std::cerr << error << '\n';
                return 1;
            }
        }
        const auto avx2_end = std::chrono::steady_clock::now();
        avx2_ms = std::chrono::duration<double, std::milli>(
            avx2_end - avx2_start).count() / iterations;
    }
#endif

    const double decode_ms = std::chrono::duration<double, std::milli>(decode_end - decode_start).count() / iterations;
    const double linear_ms = std::chrono::duration<double, std::milli>(linear_end - linear_start).count() / iterations;
    const double fused_ms = std::chrono::duration<double, std::milli>(fused_end - fused_start).count() / iterations;
    std::cout << "shape: [" << rows << ", " << cols << "]\n"
              << "payload bytes: " << bytes << "\n"
              << "decode_ms: " << decode_ms << "\n"
              << "decode_then_linear_ms: " << linear_ms << "\n"
              << "fused_linear_ms: " << fused_ms << "\n"
              << "reference_vs_fused_delta: " << (output.front() - fused_output.front()) << "\n"
              << "first output: " << output.front() << '\n';
#ifdef CHAPTER05_HAS_AVX2
    if (has_avx2) {
        std::cout << "avx2_linear_ms: " << avx2_ms << "\n"
                  << "reference_vs_avx2_delta: "
                  << (output.front() - avx2_output.front()) << '\n';
    } else {
        std::cout << "avx2_linear_ms: unavailable\n";
    }
#endif
    return 0;
}
