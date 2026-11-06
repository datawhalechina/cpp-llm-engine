// Chapter 07：声明逻辑词元到物理 K/V 块的控制面。
// 职责：按请求分配、提交、裁剪和释放固定大小物理块。
// 边界：不保存 K/V 数值，也不决定请求调度顺序。
#pragma once

#include <cstddef>
#include <unordered_map>
#include <vector>

namespace chapter07_kv_cache {

struct BlockLocation {
    std::size_t block = 0;
    std::size_t slot = 0;
};

class KvBlockPool;

class BlockTable final {
public:
    explicit BlockTable(std::size_t block_size);

    std::size_t block_size() const;
    std::size_t token_count() const;
    std::size_t block_count() const;
    std::size_t capacity_tokens() const;
    const std::vector<std::size_t>& physical_blocks() const;
    BlockLocation locate(std::size_t token_index) const;
    BlockLocation locate_reserved(std::size_t token_index) const;

private:
    friend class KvBlockPool;

    void append_block(std::size_t physical_block);
    std::size_t remove_last_block();
    void set_token_count(std::size_t token_count);

    std::size_t block_size_;
    std::size_t token_count_ = 0;
    std::vector<std::size_t> physical_blocks_;
};

class KvBlockPool final {
public:
    KvBlockPool(std::size_t total_blocks, std::size_t block_size);

    std::size_t total_blocks() const;
    std::size_t free_blocks() const;
    std::size_t used_blocks() const;
    std::size_t block_size() const;
    bool contains(std::size_t request_id) const;
    std::size_t block_count(std::size_t request_id) const;
    std::size_t token_count(std::size_t request_id) const;
    std::size_t capacity_tokens(std::size_t request_id) const;
    std::vector<std::size_t> physical_blocks(std::size_t request_id) const;

    bool reserve(std::size_t request_id, std::size_t total_tokens);
    bool commit_tokens(std::size_t request_id, std::size_t total_tokens);
    bool trim_to_tokens(std::size_t request_id, std::size_t total_tokens);
    bool release(std::size_t request_id);
    BlockLocation locate(std::size_t request_id, std::size_t token_index) const;
    BlockLocation locate_reserved(
        std::size_t request_id,
        std::size_t token_index) const;

private:
    std::size_t required_blocks(std::size_t token_count) const;

    std::size_t block_size_;
    std::vector<bool> free_;
    std::unordered_map<std::size_t, BlockTable> tables_;
};

}  // namespace chapter07_kv_cache
