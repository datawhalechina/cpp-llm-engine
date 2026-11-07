// Chapter 03：Tensor View 到矩阵乘请求的适配接口
// 输入：CPU 上连续、行主序、Float32 的二维 View，以及 CPU 计算后端。
// 输出：成功后提交 Matrix 或写入输出 View；失败时保留原输出，不隐式复制输入。
// 验证：bash code/chapter03_matmul/scripts/run.sh
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
