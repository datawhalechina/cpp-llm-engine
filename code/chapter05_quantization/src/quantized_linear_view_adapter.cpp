// Chapter 05：量化权重与公共激活 View 的适配实现。
// 职责：校验 packed 权重及激活布局，并构造单行或 batch Linear 请求。
// 契约：rank-2 交给 backend 的 batch 接口；成功后统一提交输出。
// 验证：chapter02_tensor_quantized_linear_integration_test.cpp。
#include "chapter05_quantization/quantized_linear_view_adapter.h"

#include "cpp_llm_engine/core/quantization_contract.h"

#include <algorithm>
#include <vector>

namespace chapter05_quantization {
namespace {

using cpp_llm_engine::core::DType;
using cpp_llm_engine::core::DeviceKind;
using cpp_llm_engine::core::Status;
using cpp_llm_engine::core::TensorViewRef;

Status validate_activation_view(const TensorViewRef& view) {
    const auto storage_status = view.validate(sizeof(float));
    if (!storage_status.is_ok()) return storage_status;
    const auto rank = view.contract.shape.dimensions.size();
    if (rank != 1 && rank != 2) {
        return Status::error("quantized Linear activation rank must be one or two");
    }
    if (view.contract.dtype != DType::Float32) {
        return Status::error("quantized Linear activation dtype must be Float32");
    }
    if (view.contract.device != DeviceKind::CPU) {
        return Status::error("quantized Linear activation device must be CPU");
    }
    if (!view.contract.is_contiguous_row_major()) {
        return Status::error("quantized Linear activation must be contiguous row-major");
    }
    return Status::ok();
}

}  // namespace

cpp_llm_engine::core::Status compute_quantized_linear_views(
    const QuantizedMatrixView& weights,
    const cpp_llm_engine::core::TensorViewRef& input,
    const cpp_llm_engine::core::MutableTensorViewRef& output,
    const QuantizedLinearPolicy& policy,
    const QuantizedLinearBackendRegistry& registry) {
    using cpp_llm_engine::core::LinearBatchRequest;
    using cpp_llm_engine::core::Status;

    if (weights.data == nullptr) {
        return Status::error("quantized weight data must not be null");
    }
    std::size_t expected_weight_bytes = 0;
    const auto weight_status = cpp_llm_engine::core::checked_quantized_matrix_bytes(
        weights.format, weights.rows, weights.cols, &expected_weight_bytes);
    if (!weight_status.is_ok()) return weight_status;
    if (weights.bytes != expected_weight_bytes) {
        return Status::error("quantized weight byte count does not match shape");
    }
    const auto input_status = validate_activation_view(input);
    if (!input_status.is_ok()) return input_status;
    const auto output_status = validate_activation_view(output.as_const());
    if (!output_status.is_ok()) return output_status;

    const auto& input_shape = input.contract.shape.dimensions;
    const auto& output_shape = output.contract.shape.dimensions;
    if (input_shape.size() != output_shape.size()) {
        return Status::error("quantized Linear input and output ranks must match");
    }
    const std::size_t tokens = input_shape.size() == 1 ? 1 : input_shape[0];
    const std::size_t input_width = input_shape.back();
    const std::size_t output_width = output_shape.back();
    if (input_width != weights.cols) {
        return Status::error("quantized Linear input width does not match weights");
    }
    if (output_width != weights.rows ||
        (input_shape.size() == 2 && output_shape[0] != tokens)) {
        return Status::error("quantized Linear output shape does not match weights");
    }

    std::size_t output_elements = 0;
    const auto shape_status =
        output.contract.shape.checked_numel(&output_elements);
    if (!shape_status.is_ok()) return shape_status;
    const auto* input_data = static_cast<const float*>(input.storage_data) +
                             input.contract.storage_offset_elements;
    std::vector<float> result(output_elements, 0.0f);
    const LinearBatchRequest request{
        weights.data,
        weights.bytes,
        input_data,
        tokens * weights.cols,
        result.data(),
        result.size(),
        tokens,
        weights.rows,
        weights.cols,
        weights.format,
    };
    const auto status = compute_quantized_linear_batch(request, policy, registry);
    if (!status.is_ok()) return status;

    auto* output_data = static_cast<float*>(output.storage_data) +
                        output.contract.storage_offset_elements;
    std::copy(result.begin(), result.end(), output_data);
    return Status::ok();
}

}  // namespace chapter05_quantization
