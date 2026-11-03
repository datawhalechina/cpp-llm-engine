// Chapter 02->04：N 维 Tensor 到 Attention backend 的适配实现
// 职责：把连续 [T,H,Dh] 或 [B,T,H,Dh] View 转为 AttentionRequest。
// 契约：rank、shape、dtype、device 或布局不匹配时在 backend 前失败。
// 验证：chapter02_tensor_attention_integration_test.cpp。
#include "chapter04_attention/tensor_attention_adapter.h"

#include <algorithm>
#include <utility>
#include <vector>

namespace chapter04_attention {

namespace {

using cpp_llm_engine::core::DeviceKind;
using cpp_llm_engine::core::DType;
using cpp_llm_engine::core::MutableTensorViewRef;
using cpp_llm_engine::core::Status;
using cpp_llm_engine::core::TensorViewRef;

Status validate_view(const TensorViewRef& view, std::size_t expected_rank) {
    const auto range_status = view.validate(sizeof(float));
    if (!range_status.is_ok()) return range_status;
    if (view.contract.shape.dimensions.size() != expected_rank) {
        return Status::error("attention tensor views must have the same rank 3 or 4");
    }
    if (view.contract.dtype != DType::Float32) {
        return Status::error("attention tensor view dtype must be Float32");
    }
    if (view.contract.device != DeviceKind::CPU) {
        return Status::error("attention tensor view device must be CPU");
    }
    if (!view.contract.is_contiguous_row_major()) {
        return Status::error("attention requires contiguous token-major input");
    }
    return Status::ok();
}

const float* first_element(const TensorViewRef& view) {
    return static_cast<const float*>(view.storage_data) +
           view.contract.storage_offset_elements;
}

float* first_element(const MutableTensorViewRef& view) {
    return static_cast<float*>(view.storage_data) +
           view.contract.storage_offset_elements;
}

}  // namespace

Status try_compute_tensor_attention_into(
    const TensorViewRef& query,
    const TensorViewRef& key,
    const TensorViewRef& value,
    const cpp_llm_engine::core::AttentionBackend& backend,
    const MutableTensorViewRef& output,
    std::size_t query_position_offset,
    bool causal,
    float scale) {
    const auto rank = query.contract.shape.dimensions.size();
    if (rank != 3 && rank != 4) {
        return Status::error("attention tensor rank must be 3 or 4");
    }
    for (const auto* view : {&query, &key, &value}) {
        const auto status = validate_view(*view, rank);
        if (!status.is_ok()) return status;
    }
    const auto output_status = validate_view(output.as_const(), rank);
    if (!output_status.is_ok()) return output_status;

    const auto& q_shape = query.contract.shape.dimensions;
    const auto& k_shape = key.contract.shape.dimensions;
    const auto& v_shape = value.contract.shape.dimensions;
    const std::size_t batch_axis = 0;
    const std::size_t token_axis = rank == 3 ? 0 : 1;
    const std::size_t head_axis = rank == 3 ? 1 : 2;
    const std::size_t dim_axis = rank == 3 ? 2 : 3;
    const std::size_t batch = rank == 3 ? 1 : q_shape[batch_axis];

    if ((rank == 4 && (k_shape[batch_axis] != batch || v_shape[batch_axis] != batch)) ||
        q_shape[head_axis] != k_shape[head_axis] ||
        q_shape[head_axis] != v_shape[head_axis] ||
        q_shape[dim_axis] != k_shape[dim_axis] ||
        q_shape[dim_axis] != v_shape[dim_axis]) {
        return Status::error("attention batch, heads, and head_dim must match");
    }
    if (k_shape[token_axis] != v_shape[token_axis]) {
        return Status::error("attention key and value token counts must match");
    }
    if (output.contract.shape.dimensions != q_shape) {
        return Status::error("attention output shape must match query");
    }

    std::size_t query_elements = 0;
    auto shape_status = query.contract.shape.checked_numel(&query_elements);
    if (!shape_status.is_ok()) return shape_status;
    std::size_t key_elements = 0;
    shape_status = key.contract.shape.checked_numel(&key_elements);
    if (!shape_status.is_ok()) return shape_status;
    std::vector<float> result(query_elements, 0.0f);

    const std::size_t query_batch_elements = query_elements / batch;
    const std::size_t key_batch_elements = key_elements / batch;
    const float* query_data = first_element(query);
    const float* key_data = first_element(key);
    const float* value_data = first_element(value);

    for (std::size_t batch_index = 0; batch_index < batch; ++batch_index) {
        const cpp_llm_engine::core::AttentionRequest request{
            query_data + batch_index * query_batch_elements,
            key_data + batch_index * key_batch_elements,
            value_data + batch_index * key_batch_elements,
            result.data() + batch_index * query_batch_elements,
            q_shape[token_axis],
            k_shape[token_axis],
            q_shape[head_axis],
            q_shape[dim_axis],
            query_position_offset,
            causal,
            scale,
        };
        const auto status = backend.compute(request);
        if (!status.is_ok()) return status;
    }

    std::copy(result.begin(), result.end(), first_element(output));
    return Status::ok();
}

Status try_compute_tensor_attention(
    const TensorViewRef& query,
    const TensorViewRef& key,
    const TensorViewRef& value,
    const cpp_llm_engine::core::AttentionBackend& backend,
    chapter02_tensor_ops::Tensor* output,
    std::size_t query_position_offset,
    bool causal,
    float scale) {
    if (output == nullptr) return Status::error("attention output must not be null");
    chapter02_tensor_ops::Tensor result(query.contract.shape.dimensions);
    const auto status = try_compute_tensor_attention_into(
        query, key, value, backend, result.mutable_view_ref(),
        query_position_offset, causal, scale);
    if (!status.is_ok()) return status;
    *output = std::move(result);
    return Status::ok();
}

}  // namespace chapter04_attention
