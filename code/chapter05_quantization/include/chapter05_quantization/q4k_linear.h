// 中文：提供 Q4_K Linear 的 reference、scalar fused 和 AVX2 backend 接口。
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "cpp_llm_engine/core/linear_contract.h"

namespace chapter05_quantization {

class Q4KLinearReference final {
public:
    // Treats weights as row-major [rows, cols] Q4_K blocks and computes W * input.
    static bool forward(const std::uint8_t* weights,
                        std::size_t weight_bytes,
                        std::size_t rows,
                        std::size_t cols,
                        const std::vector<float>& input,
                        std::vector<float>* output,
                        std::string* error = nullptr);
};

class Q4KLinearFused final {
public:
    // Decodes each Q4_K block directly into the dot-product accumulator.
    static bool forward(const std::uint8_t* weights,
                        std::size_t weight_bytes,
                        std::size_t rows,
                        std::size_t cols,
                        const std::vector<float>& input,
                        std::vector<float>* output,
                        std::string* error = nullptr);
};

class Q4KLinearFusedBackend final : public cpp_llm_engine::core::LinearBackend {
public:
    cpp_llm_engine::core::BackendKind kind() const override;
    std::string_view name() const override;
    cpp_llm_engine::core::Status compute(
        const cpp_llm_engine::core::LinearRequest& request) const override;
    cpp_llm_engine::core::Status compute_batch(
        const cpp_llm_engine::core::LinearBatchRequest& request) const override;
};

const Q4KLinearFusedBackend& q4k_fused_backend();

class Q4KLinearAvx2 final {
public:
    static bool forward(const std::uint8_t* weights,
                        std::size_t weight_bytes,
                        std::size_t rows,
                        std::size_t cols,
                        const std::vector<float>& input,
                        std::vector<float>* output,
                        std::string* error = nullptr);
};

class Q4KLinearAvx2Backend final : public cpp_llm_engine::core::LinearBackend {
public:
    cpp_llm_engine::core::BackendKind kind() const override;
    std::string_view name() const override;
    cpp_llm_engine::core::Status compute(
        const cpp_llm_engine::core::LinearRequest& request) const override;
};

const Q4KLinearAvx2Backend& q4k_avx2_backend();

// AVX2 backend 还依赖 FMA；调用专用实现前先检查运行机器能力。
bool q4k_avx2_supported();

}  // namespace chapter05_quantization
