// Chapter 07：声明按词元前缀索引完整 K/V 块的 Radix Cache。
// 职责：最长前缀匹配、共享块引用、叶端淘汰和新请求页表挂接。
// 限制：只共享块对齐前缀，不提供跨线程锁或调度排序。
#pragma once

#include <cstddef>
#include <memory>
#include <unordered_map>
#include <vector>

namespace chapter07_kv_cache {

struct PrefixMatch {
    std::size_t token_count = 0;
    std::vector<std::size_t> physical_blocks;
};

struct RadixPrefixCacheStats {
    std::size_t queries = 0;
    std::size_t hits = 0;
    std::size_t misses = 0;
    std::size_t matched_tokens = 0;
    std::size_t evicted_blocks = 0;
};

// Radix 索引只依赖块所有权契约，CPU/CUDA 分别负责清理自己的 payload。
class PrefixBlockStore {
public:
    virtual ~PrefixBlockStore() = default;

    virtual std::size_t total_blocks() const = 0;
    virtual std::size_t free_blocks() const = 0;
    virtual std::size_t block_size() const = 0;
    virtual std::size_t token_count(std::size_t request_id) const = 0;
    virtual std::vector<std::size_t> physical_blocks(
        std::size_t request_id) const = 0;
    virtual std::size_t block_ref_count(std::size_t physical_block) const = 0;
    virtual bool attach_shared_prefix(
        std::size_t request_id,
        const std::vector<std::size_t>& physical_blocks,
        std::size_t token_count) = 0;
    virtual bool retain_prefix_blocks(
        const std::vector<std::size_t>& physical_blocks) = 0;
    virtual bool release_prefix_blocks(
        const std::vector<std::size_t>& physical_blocks) = 0;
};

class RadixPrefixCache final {
public:
    explicit RadixPrefixCache(PrefixBlockStore* store);
    ~RadixPrefixCache();

    RadixPrefixCache(const RadixPrefixCache&) = delete;
    RadixPrefixCache& operator=(const RadixPrefixCache&) = delete;

    bool publish(
        const std::vector<int>& tokens,
        std::size_t request_id);
    std::size_t estimate_prefix_tokens(const std::vector<int>& tokens) const;
    PrefixMatch longest_prefix(const std::vector<int>& tokens) const;
    bool attach(
        std::size_t request_id,
        const PrefixMatch& match);
    bool evict_until_free(std::size_t required_free_blocks);
    void clear();
    std::size_t cached_blocks() const;
    RadixPrefixCacheStats stats() const;

private:
    struct Node {
        std::unordered_map<int, std::unique_ptr<Node>> children;
        bool has_physical_block = false;
        std::size_t physical_block = 0;
        mutable std::size_t last_access = 0;
    };

    bool find_eviction_candidate(
        Node* node,
        Node** candidate,
        std::size_t* oldest_access) const;
    bool evict_one();
    void release_node(Node* node);

    PrefixBlockStore* store_;
    Node root_;
    std::size_t cached_blocks_ = 0;
    mutable std::size_t access_clock_ = 0;
    mutable RadixPrefixCacheStats stats_;
};

}  // namespace chapter07_kv_cache
