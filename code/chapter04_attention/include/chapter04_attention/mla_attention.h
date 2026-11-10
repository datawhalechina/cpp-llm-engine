// Chapter 04：多头潜在注意力（MLA）CPU 数值参考接口。
// 职责：从低秩 K/V 缓存临时展开各头 K/V，并执行因果 Attention。
// 契约：缓存保持压缩布局；失败时不覆盖调用方输出。
// 验证：mla_attention_test.cpp 与显式展开的多头 Attention golden 对照。
#pragma once

#include "cpp_llm_engine/core/status.h"

#include <cstddef>

namespace chapter04_attention {

struct MlaAttentionRequest {
    // [query_tokens, heads, qk_nope_dim]
    const float* query_nope = nullptr;
    // [query_tokens, heads, rope_dim]，调用前已经应用 RoPE。
    const float* query_rope = nullptr;
    // [key_tokens, latent_dim]，请求级持久缓存的数据部分。
    const float* latent_kv = nullptr;
    // [key_tokens, rope_dim]，请求级持久缓存的位置部分。
    const float* key_rope = nullptr;
    // [heads, qk_nope_dim, latent_dim]
    const float* key_up_weight = nullptr;
    // [heads, value_dim, latent_dim]
    const float* value_up_weight = nullptr;
    // [query_tokens, heads, value_dim]
    float* output = nullptr;

    std::size_t query_tokens = 0;
    std::size_t key_tokens = 0;
    std::size_t query_position_offset = 0;
    std::size_t heads = 0;
    std::size_t qk_nope_dim = 0;
    std::size_t rope_dim = 0;
    std::size_t latent_dim = 0;
    std::size_t value_dim = 0;
    bool causal = true;
    // 0 表示 1 / sqrt(qk_nope_dim + rope_dim)。
    float scale = 0.0F;
};

cpp_llm_engine::core::Status cpu_mla_attention_reference(
    const MlaAttentionRequest& request);

}  // namespace chapter04_attention
