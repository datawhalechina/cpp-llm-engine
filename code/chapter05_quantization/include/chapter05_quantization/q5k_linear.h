// Chapter 05：声明 GGML Q5_K 的 scalar CPU Linear backend。
// 职责：公开 Q5_K block 常量和统一 Linear 接口实现。
// 契约：权重按 row-major 保存，每 256 个值使用一个 176-byte super-block。
// 验证：chapter05_quantization_test 的 Q5_K 固定 golden。
#pragma once
#include "cpp_llm_engine/core/linear_contract.h"
#include <cstddef>
#include <cstdint>
#include <string_view>
namespace chapter05_quantization {
constexpr std::size_t kQ5KBlockElements = 256;
constexpr std::size_t kQ5KBlockBytes = 176;
class Q5KLinearBackend final : public cpp_llm_engine::core::LinearBackend {
public:
    cpp_llm_engine::core::BackendKind kind() const override;
    std::string_view name() const override;
    cpp_llm_engine::core::Status compute(const cpp_llm_engine::core::LinearRequest&) const override;
    cpp_llm_engine::core::Status compute_batch(
        const cpp_llm_engine::core::LinearBatchRequest&) const override;
};
const Q5KLinearBackend& q5k_backend();
}
