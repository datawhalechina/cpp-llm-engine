// Chapter 04：单层 Decoder 请求上下文
// 保存一个请求在当前层的连续 K/V；权重仍由 DecoderLayer 共享。
// 状态只在 prefill/decode 全部成功后提交，分页存储由 Chapter07 负责。
// 验证：chapter04_attention_test.cpp。
#pragma once

#include "chapter04_attention/kv_cache.h"

#include <cstddef>

namespace chapter04_attention {

class DecoderLayer;

class DecoderLayerContext final {
public:
    DecoderLayerContext() = default;
    DecoderLayerContext(
        std::size_t capacity,
        std::size_t heads,
        std::size_t head_dim);

    bool ready() const;
    void reset();

    std::size_t token_count() const;
    std::size_t capacity() const;
    const KvCache& kv_cache() const;

private:
    friend class DecoderLayer;
    KvCache kv_cache_;
};

}  // namespace chapter04_attention
