// Chapter 03：Tensor View 到 MatMul 请求的适配接口
// 职责：校验 CPU Float32 连续二维 View，并交给指定 MatMul backend。
// 契约：不拥有 Storage；只接受 CPU Float32 连续二维 View 和 CPU backend；失败保留输出。
// 验证：chapter02_tensor_matmul_integration_test.cpp。
#pragma once

#include "chapter03_matmul/matrix.h"
#include "cpp_llm_engine/core/matmul_contract.h"
#include "cpp_llm_engine/core/tensor_view_ref.h"

namespace chapter03_matmul {

cpp_llm_engine::core::Status try_compute_tensor_views_into(
    const cpp_llm_engine::core::TensorViewRef& lhs,
    const cpp_llm_engine::core::TensorViewRef& rhs,
    const cpp_llm_engine::core::MatmulBackend& backend,
    const cpp_llm_engine::core::MutableTensorViewRef& output);

cpp_llm_engine::core::Status try_compute_tensor_views(
    const cpp_llm_engine::core::TensorViewRef& lhs,
    const cpp_llm_engine::core::TensorViewRef& rhs,
    const cpp_llm_engine::core::MatmulBackend& backend,
    Matrix* output);

}  // namespace chapter03_matmul
