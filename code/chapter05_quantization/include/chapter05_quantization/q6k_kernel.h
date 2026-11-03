// 中文：声明 Q6_K fused CPU kernel；直接从 block payload 累加 Linear 结果。
#pragma once

#include "cpp_llm_engine/core/linear_contract.h"

namespace chapter05_quantization {

// 校验 Q6_K Linear 请求，并在不创建 float 权重 buffer 的情况下计算 W * input。
cpp_llm_engine::core::Status q6k_fused_cpu_kernel(
    const cpp_llm_engine::core::LinearRequest& request);
cpp_llm_engine::core::Status q6k_fused_cpu_batch_kernel(
    const cpp_llm_engine::core::LinearBatchRequest& request);

}  // namespace chapter05_quantization
