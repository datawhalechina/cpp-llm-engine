// Chapter 03：CPU 矩阵乘参考内核接口
// 输入输出：计算行主序 [M,K] * [K,N]，结果写入调用方提供的 [M,N] 缓冲区。
// 限制：内核不拥有缓冲区，也不分配 Matrix；校验失败时返回 Status。
// 验证：bash code/chapter03_matmul/scripts/run.sh
#pragma once

#include "cpp_llm_engine/core/matmul_contract.h"

namespace chapter03_matmul {

cpp_llm_engine::core::Status cpu_reference_matmul_kernel(
    const cpp_llm_engine::core::MatmulRequest& request);

}  // namespace chapter03_matmul
