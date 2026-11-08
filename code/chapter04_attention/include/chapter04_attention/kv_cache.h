// Chapter 04：固定容量 KV Cache 接口
// 职责：保存按 [tokens, heads, head_dim] 排列的历史 key/value。
// 追加接口同时接收缓冲区元素数，拒绝空指针、短缓冲区和容量溢出。
// 验证：chapter04_attention_test.cpp。
#pragma once

#include "cpp_llm_engine/core/status.h"

#include <cstddef>
#include <vector>

namespace chapter04_attention {

class KvCache final {
public:
    KvCache() = default;
    KvCache(std::size_t capacity, std::size_t heads, std::size_t head_dim);

    bool ready() const;
    cpp_llm_engine::core::Status try_append(
        const float* key,
        std::size_t key_element_count,
        const float* value,
        std::size_t value_element_count,
        std::size_t token_count);
    void clear();

    std::size_t size() const;
    std::size_t capacity() const;
    std::size_t heads() const;
    std::size_t head_dim() const;
    std::size_t width() const;
    const float* key_data() const;
    const float* value_data() const;

private:
    std::size_t capacity_ = 0;
    std::size_t heads_ = 0;
    std::size_t head_dim_ = 0;
    std::size_t size_ = 0;
    std::vector<float> key_;
    std::vector<float> value_;
};

}  // namespace chapter04_attention
