// Chapter 02->04：N 维 Tensor 到 Attention backend 的适配接口
// 职责：验证 3D/4D Q/K/V 布局，并按 batch 生成 AttentionRequest。
// 契约：只接受连续 CPU Float32 View，不执行隐式连续化。
// 验证：chapter02_tensor_attention_integration_test.cpp。
#pragma once

#include "chapter02_tensor_ops/tensor.h"
#include "cpp_llm_engine/core/attention_contract.h"
#include "cpp_llm_engine/core/tensor_view_ref.h"

#include <cstddef>

namespace chapter04_attention {

cpp_llm_engine::core::Status try_compute_tensor_attention_into(
    const cpp_llm_engine::core::TensorViewRef& query,
    const cpp_llm_engine::core::TensorViewRef& key,
    const cpp_llm_engine::core::TensorViewRef& value,
    const cpp_llm_engine::core::AttentionBackend& backend,
    const cpp_llm_engine::core::MutableTensorViewRef& output,
    std::size_t query_position_offset = 0,
    bool causal = true,
    float scale = 0.0f);

cpp_llm_engine::core::Status try_compute_tensor_attention(
    const cpp_llm_engine::core::TensorViewRef& query,
    const cpp_llm_engine::core::TensorViewRef& key,
    const cpp_llm_engine::core::TensorViewRef& value,
    const cpp_llm_engine::core::AttentionBackend& backend,
    chapter02_tensor_ops::Tensor* output,
    std::size_t query_position_offset = 0,
    bool causal = true,
    float scale = 0.0f);

}  // namespace chapter04_attention
