// Chapter 05：声明 Q6_K fused CPU kernel。
// 实现直接从 block payload 累加 Linear 结果，不创建完整浮点权重。
// 验证入口：chapter05_quantization_test。
#pragma once

#include "cpp_llm_engine/core/linear_contract.h"

namespace chapter05_quantization {

// 校验 Q6_K Linear 请求，并在不创建 float 权重 buffer 的情况下计算 W * input。
cpp_llm_engine::core::Status q6k_fused_cpu_kernel(
    const cpp_llm_engine::core::LinearRequest& request);
cpp_llm_engine::core::Status q6k_fused_cpu_batch_kernel(
    const cpp_llm_engine::core::LinearBatchRequest& request);

}  // namespace chapter05_quantization
