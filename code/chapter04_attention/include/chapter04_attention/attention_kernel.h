// Chapter 04：CPU Attention kernel 接口
// 职责：声明 row-major Q/K/V 的三段式 golden 与分块在线 Softmax kernel。
// 契约：调用方拥有 buffer，非法尺寸或不可见位置返回 Status 错误。
// 验证：chapter04_attention_test.cpp、online_softmax_test.cpp。
#pragma once

#include "cpp_llm_engine/core/attention_contract.h"

namespace chapter04_attention {

// Row-major CPU reference attention with optional causal masking.
cpp_llm_engine::core::Status cpu_reference_attention_kernel(
    const cpp_llm_engine::core::AttentionRequest& request);

// CPU 数学参考：按 key tile 合并 max、指数和与 Value 累加状态。
// 该函数验证 FlashAttention 的在线 Softmax 基础，不代表 GPU FlashAttention 实现。
cpp_llm_engine::core::Status cpu_online_attention_kernel(
    const cpp_llm_engine::core::AttentionRequest& request,
    std::size_t tile_size);

}  // namespace chapter04_attention
