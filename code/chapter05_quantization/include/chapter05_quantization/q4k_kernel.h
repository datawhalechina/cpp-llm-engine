// Chapter 05：声明 Q4_K scalar fused Linear kernel。
// 职责：直接解码 packed block 并累加，避免完整浮点权重缓冲区。
#pragma once

#include "cpp_llm_engine/core/linear_contract.h"

namespace chapter05_quantization {

// Validates a Q4_K Linear request and computes W * input without a float weight buffer.
cpp_llm_engine::core::Status q4k_fused_cpu_kernel(
    const cpp_llm_engine::core::LinearRequest& request);
cpp_llm_engine::core::Status q4k_fused_cpu_batch_kernel(
    const cpp_llm_engine::core::LinearBatchRequest& request);

}  // namespace chapter05_quantization
