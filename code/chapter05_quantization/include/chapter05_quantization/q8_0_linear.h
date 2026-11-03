// Chapter 05：Q8_0 Linear 接口
// 职责：提供完整解码 reference、直接 decode-and-dot fused 与统一 CPU backend。
// 契约：权重按 row-major [rows, cols] 保存，每 32 个值使用一个 34-byte block。
// 验证：chapter05_quantization_test 与 SmolVLM Q8_0 Decoder 路径。
#pragma once

#include "cpp_llm_engine/core/linear_contract.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace chapter05_quantization {

class Q8_0LinearReference final {
public:
    static bool forward(const std::uint8_t* weights,
                        std::size_t weight_bytes,
                        std::size_t rows,
                        std::size_t cols,
                        const std::vector<float>& input,
                        std::vector<float>* output,
                        std::string* error = nullptr);
};

class Q8_0LinearFused final {
public:
    static bool forward(const std::uint8_t* weights,
                        std::size_t weight_bytes,
                        std::size_t rows,
                        std::size_t cols,
                        const std::vector<float>& input,
                        std::vector<float>* output,
                        std::string* error = nullptr);
};

class Q8_0LinearFusedBackend final : public cpp_llm_engine::core::LinearBackend {
public:
    cpp_llm_engine::core::BackendKind kind() const override;
    std::string_view name() const override;
    cpp_llm_engine::core::Status compute(
        const cpp_llm_engine::core::LinearRequest& request) const override;
    cpp_llm_engine::core::Status compute_batch(
        const cpp_llm_engine::core::LinearBatchRequest& request) const override;
};

const Q8_0LinearFusedBackend& q8_0_fused_backend();

}  // namespace chapter05_quantization
