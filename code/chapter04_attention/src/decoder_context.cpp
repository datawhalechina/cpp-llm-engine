// Chapter 04：单层 Decoder 请求上下文实现
// 将位置直接表示为连续 KV Cache 的 token 数，reset 后可复用 storage。
// Context 不持有 Decoder 权重，也不负责分页分配或跨请求调度。
// 验证：chapter04_attention_test.cpp。
#include "chapter04_attention/decoder_context.h"

namespace chapter04_attention {

DecoderLayerContext::DecoderLayerContext(
    std::size_t capacity,
    std::size_t heads,
    std::size_t head_dim)
    : kv_cache_(capacity, heads, head_dim) {}

bool DecoderLayerContext::ready() const {
    return kv_cache_.ready();
}

void DecoderLayerContext::reset() {
    kv_cache_.clear();
}

std::size_t DecoderLayerContext::token_count() const {
    return kv_cache_.size();
}

std::size_t DecoderLayerContext::capacity() const {
    return kv_cache_.capacity();
}

const KvCache& DecoderLayerContext::kv_cache() const {
    return kv_cache_;
}

}  // namespace chapter04_attention
