// Chapter 05：量化权重与公共激活 View 的适配接口。
// 职责：将 rank-1/2 Float32 激活派发给量化 Linear 的单行或 batch backend。
// 契约：packed 权重保持专用描述；激活连续且位于 CPU；失败不修改输出。
// 验证：chapter02_tensor_quantized_linear_integration_test.cpp。
#pragma once

#include "chapter05_quantization/quantized_linear_policy.h"
#include "cpp_llm_engine/core/tensor_view_ref.h"

#include <cstddef>

namespace chapter05_quantization {

struct QuantizedMatrixView final {
    const void* data = nullptr;
    std::size_t bytes = 0;
    std::size_t rows = 0;
    std::size_t cols = 0;
    cpp_llm_engine::core::WeightFormat format =
        cpp_llm_engine::core::WeightFormat::Unknown;
};

cpp_llm_engine::core::Status compute_quantized_linear_views(
    const QuantizedMatrixView& weights,
    const cpp_llm_engine::core::TensorViewRef& input,
    const cpp_llm_engine::core::MutableTensorViewRef& output,
    const QuantizedLinearPolicy& policy = {},
    const QuantizedLinearBackendRegistry& registry = {});

}  // namespace chapter05_quantization
