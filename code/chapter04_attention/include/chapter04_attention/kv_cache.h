// Chapter 04：固定容量 KV Cache 接口
// 职责：保存按 [tokens, heads, head_dim] 排列的历史 key/value。
// 契约：append 不能超过 capacity，clear 只重置 token 数并保留 storage。
// 验证：chapter04_attention_test.cpp。
#pragma once

#include <cstddef>
#include <vector>

namespace chapter04_attention {

class KvCache final {
public:
    KvCache() = default;
    KvCache(std::size_t capacity, std::size_t heads, std::size_t head_dim);

    bool ready() const;
    bool append(const float* key, const float* value, std::size_t token_count);
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
