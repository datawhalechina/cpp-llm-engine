// Core：RoPE backend 请求契约
// 职责：固定原地旋转的 token、head、head_dim、位置和频率参数。
// 契约：values 使用 [tokens, heads, head_dim] 连续布局，只旋转前 rotary_dimension。
// 验证：Core contract test 与 CUDA RoPE CPU golden 对照。
#pragma once

#include "status.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace cpp_llm_engine::core {

enum class RotaryPairing : std::uint8_t {
    Adjacent,
    SplitHalf,
};

struct RotaryEmbeddingRequest final {
    float* values = nullptr;
    std::size_t token_count = 0;
    std::size_t head_count = 0;
    std::size_t head_dimension = 0;
    std::size_t rotary_dimension = 0;
    std::size_t position_offset = 0;
    float frequency_base = 10000.0F;
    RotaryPairing pairing = RotaryPairing::Adjacent;

    Status validate() const {
        if (token_count == 0) return Status::ok();
        if (values == nullptr || head_count == 0 || head_dimension == 0 ||
            rotary_dimension == 0 || rotary_dimension > head_dimension ||
            rotary_dimension % 2 != 0) {
            return Status::error("RoPE request has invalid pointer or dimensions");
        }
        if (!std::isfinite(frequency_base) || frequency_base <= 0.0F) {
            return Status::error("RoPE frequency base must be finite and positive");
        }
        if (pairing != RotaryPairing::Adjacent &&
            pairing != RotaryPairing::SplitHalf) {
            return Status::error("RoPE pairing is invalid");
        }
        if (position_offset > std::numeric_limits<std::size_t>::max() -
                                  (token_count - 1)) {
            return Status::error("RoPE position range overflows");
        }
        if (head_count > std::numeric_limits<std::size_t>::max() / head_dimension ||
            token_count > std::numeric_limits<std::size_t>::max() /
                              (head_count * head_dimension)) {
            return Status::error("RoPE element count overflows");
        }
        return Status::ok();
    }
};

struct RotaryEmbeddingBatchRequest final {
    float* values = nullptr;
    const std::size_t* positions = nullptr;
    std::size_t batch_size = 0;
    std::size_t head_count = 0;
    std::size_t head_dimension = 0;
    std::size_t rotary_dimension = 0;
    float frequency_base = 10000.0F;
    RotaryPairing pairing = RotaryPairing::Adjacent;

    Status validate() const {
        if (values == nullptr || positions == nullptr || batch_size == 0 ||
            head_count == 0 || head_dimension == 0 || rotary_dimension == 0 ||
            rotary_dimension > head_dimension || rotary_dimension % 2 != 0) {
            return Status::error("batch RoPE request has invalid pointer or dimensions");
        }
        if (!std::isfinite(frequency_base) || frequency_base <= 0.0F) {
            return Status::error("batch RoPE frequency base must be finite and positive");
        }
        if (pairing != RotaryPairing::Adjacent &&
            pairing != RotaryPairing::SplitHalf) {
            return Status::error("batch RoPE pairing is invalid");
        }
        if (head_count > std::numeric_limits<std::size_t>::max() / head_dimension ||
            batch_size > std::numeric_limits<std::size_t>::max() /
                             (head_count * head_dimension)) {
            return Status::error("batch RoPE element count overflows");
        }
        return Status::ok();
    }
};

}  // namespace cpp_llm_engine::core
