// Core：Decoder 逐元素算子请求契约。
// 职责：固定 RMSNorm 与 SwiGLU 的连续 Float32 输入输出和数值参数。
// 契约：backend 不拥有 Tensor；RMSNorm weight 与输入等宽，SwiGLU 两路等宽。
// 验证：Core contract test 与 CUDA CPU golden 对照。
#pragma once

#include "status.h"

#include <cmath>
#include <cstddef>

namespace cpp_llm_engine::core {

struct RmsNormRequest final {
    const float* input = nullptr;
    const float* weight = nullptr;
    float* output = nullptr;
    std::size_t elements = 0;
    float epsilon = 0.0F;

    Status validate() const {
        if (input == nullptr || weight == nullptr || output == nullptr || elements == 0) {
            return Status::error("RMSNorm buffers and element count are invalid");
        }
        if (!std::isfinite(epsilon) || epsilon <= 0.0F) {
            return Status::error("RMSNorm epsilon must be finite and positive");
        }
        return Status::ok();
    }
};

struct RmsNormBatchRequest final {
    const float* input = nullptr;
    const float* weight = nullptr;
    float* output = nullptr;
    std::size_t rows = 0;
    std::size_t width = 0;
    float epsilon = 0.0F;

    Status validate() const {
        if (input == nullptr || weight == nullptr || output == nullptr ||
            rows == 0 || width == 0) {
            return Status::error("batch RMSNorm buffers and shape are invalid");
        }
        if (!std::isfinite(epsilon) || epsilon <= 0.0F) {
            return Status::error("batch RMSNorm epsilon must be finite and positive");
        }
        return Status::ok();
    }
};

struct SwiGluRequest final {
    const float* gate = nullptr;
    const float* up = nullptr;
    float* output = nullptr;
    std::size_t elements = 0;

    Status validate() const {
        if (gate == nullptr || up == nullptr || output == nullptr || elements == 0) {
            return Status::error("SwiGLU buffers and element count are invalid");
        }
        return Status::ok();
    }
};

}  // namespace cpp_llm_engine::core
