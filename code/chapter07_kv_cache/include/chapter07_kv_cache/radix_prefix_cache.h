// Chapter 07：声明按词元前缀索引完整 K/V 块的 CPU Radix Cache。
// 职责：最长前缀匹配、共享块引用和新请求页表挂接。
// 限制：只共享块对齐前缀，不负责淘汰、并发锁和调度排序。
#pragma once

#include "chapter07_kv_cache/paged_kv_cache.h"

#include <cstddef>
#include <memory>
#include <unordered_map>
#include <vector>

namespace chapter07_kv_cache {

struct PrefixMatch {
    std::size_t token_count = 0;
    std::vector<std::size_t> physical_blocks;
};

class RadixPrefixCache final {
public:
    explicit RadixPrefixCache(CpuPagedKvCache* cache);
    ~RadixPrefixCache();

    RadixPrefixCache(const RadixPrefixCache&) = delete;
    RadixPrefixCache& operator=(const RadixPrefixCache&) = delete;

    bool publish(
        const std::vector<int>& tokens,
        std::size_t request_id);
    PrefixMatch longest_prefix(const std::vector<int>& tokens) const;
    bool attach(
        std::size_t request_id,
        const PrefixMatch& match);
    void clear();
    std::size_t cached_blocks() const;

private:
    struct Node {
        std::unordered_map<int, std::unique_ptr<Node>> children;
        bool has_physical_block = false;
        std::size_t physical_block = 0;
    };

    void release_node(Node* node);

    CpuPagedKvCache* cache_;
    Node root_;
    std::size_t cached_blocks_ = 0;
};

}  // namespace chapter07_kv_cache
