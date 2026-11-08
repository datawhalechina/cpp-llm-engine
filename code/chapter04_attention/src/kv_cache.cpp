// Chapter 04：固定容量 KV Cache 实现
// 职责：分配连续 K/V storage，追加 token 并提供只读数据指针。
// 契约：容量、head 数和 head_dim 溢出时对象无效，append 拒绝超容量写入。
// 验证：chapter04_attention_test.cpp。
#include "chapter04_attention/kv_cache.h"

#include <algorithm>
#include <limits>

namespace chapter04_attention {

KvCache::KvCache(std::size_t capacity, std::size_t heads, std::size_t head_dim)
    : capacity_(capacity), heads_(heads), head_dim_(head_dim) {
    if (capacity_ == 0 || heads_ == 0 || head_dim_ == 0 ||
        heads_ > std::numeric_limits<std::size_t>::max() / head_dim_ ||
        capacity_ > std::numeric_limits<std::size_t>::max() / (heads_ * head_dim_)) {
        capacity_ = 0;
        heads_ = 0;
        head_dim_ = 0;
        return;
    }
    const std::size_t element_count = capacity_ * heads_ * head_dim_;
    key_.resize(element_count, 0.0f);
    value_.resize(element_count, 0.0f);
}

bool KvCache::ready() const {
    return capacity_ > 0 && heads_ > 0 && head_dim_ > 0 && key_.size() == value_.size() &&
           key_.size() == capacity_ * heads_ * head_dim_;
}

bool KvCache::append(const float* key, const float* value, std::size_t token_count) {
    if (!ready() || key == nullptr || value == nullptr || token_count == 0 ||
        token_count > capacity_ - size_) {
        return false;
    }
    const std::size_t width = heads_ * head_dim_;
    const std::size_t source_count = token_count * width;
    const std::size_t destination_offset = size_ * width;
    std::copy_n(key, source_count, key_.begin() + destination_offset);
    std::copy_n(value, source_count, value_.begin() + destination_offset);
    size_ += token_count;
    return true;
}

void KvCache::clear() {
    size_ = 0;
}

std::size_t KvCache::size() const { return size_; }
std::size_t KvCache::capacity() const { return capacity_; }
std::size_t KvCache::heads() const { return heads_; }
std::size_t KvCache::head_dim() const { return head_dim_; }
std::size_t KvCache::width() const { return heads_ * head_dim_; }
const float* KvCache::key_data() const { return ready() ? key_.data() : nullptr; }
const float* KvCache::value_data() const { return ready() ? value_.data() : nullptr; }

}  // namespace chapter04_attention
