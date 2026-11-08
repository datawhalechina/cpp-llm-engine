// Chapter 07：声明 CPU 分页 K/V 数据面和分页 Attention 参考实现。
// 职责：按 layer/block/slot/head/dimension 保存真实 K/V 数值。
// 边界：不拥有 Request、Scheduler 或网络服务状态。
#pragma once

#include "chapter07_kv_cache/block_table.h"

#include <cstddef>
#include <string>
#include <utility>
#include <vector>

namespace chapter07_kv_cache {

class CpuPagedKvCache final : public KvCacheTransaction {
public:
    CpuPagedKvCache(
        std::size_t layer_count,
        std::size_t total_blocks,
        std::size_t block_size,
        std::size_t kv_heads,
        std::size_t head_dim);

    std::size_t layer_count() const;
    std::size_t total_blocks() const override;
    std::size_t free_blocks() const override;
    std::size_t used_blocks() const override;
    std::size_t block_size() const override;
    std::size_t kv_heads() const;
    std::size_t head_dim() const;
    std::size_t payload_bytes() const;

    const KvBlockPool& block_pool() const;
    bool contains(std::size_t request_id) const override;
    std::size_t block_count(std::size_t request_id) const override;
    std::size_t token_count(std::size_t request_id) const override;
    std::size_t capacity_tokens(std::size_t request_id) const override;
    std::vector<std::size_t> physical_blocks(
        std::size_t request_id) const override;
    BlockLocation locate(
        std::size_t request_id,
        std::size_t token_index) const override;
    BlockLocation locate_reserved(
        std::size_t request_id,
        std::size_t token_index) const override;

    bool reserve(std::size_t request_id, std::size_t total_tokens) override;
    bool attach_shared_prefix(
        std::size_t request_id,
        const std::vector<std::size_t>& physical_blocks,
        std::size_t token_count);
    bool retain_prefix_blocks(
        const std::vector<std::size_t>& physical_blocks);
    bool release_prefix_blocks(
        const std::vector<std::size_t>& physical_blocks);
    std::size_t block_ref_count(std::size_t physical_block) const;
    bool write_token(
        std::size_t request_id,
        std::size_t layer,
        std::size_t token_index,
        const float* key,
        const float* value,
        std::size_t element_count,
        std::string* error = nullptr);
    bool commit_tokens(
        std::size_t request_id,
        std::size_t total_tokens) override;
    bool commit_tokens(
        std::size_t request_id,
        std::size_t total_tokens,
        std::string* error);
    bool commit_batch(
        const std::vector<std::pair<std::size_t, std::size_t>>& requests,
        std::string* error = nullptr);
    bool trim_to_tokens(
        std::size_t request_id,
        std::size_t total_tokens) override;
    bool release(std::size_t request_id) override;

    const float* key_data(
        std::size_t request_id,
        std::size_t layer,
        std::size_t token_index,
        std::size_t kv_head) const;
    const float* value_data(
        std::size_t request_id,
        std::size_t layer,
        std::size_t token_index,
        std::size_t kv_head) const;
    const float* key_data_reserved(
        std::size_t request_id,
        std::size_t layer,
        std::size_t token_index,
        std::size_t kv_head) const;
    const float* value_data_reserved(
        std::size_t request_id,
        std::size_t layer,
        std::size_t token_index,
        std::size_t kv_head) const;
    bool attention(
        std::size_t request_id,
        std::size_t layer,
        const std::vector<float>& query,
        std::size_t query_heads,
        std::vector<float>* output,
        std::string* error = nullptr) const;

private:
    std::size_t payload_offset(
        std::size_t layer,
        BlockLocation location,
        std::size_t kv_head) const;
    std::size_t written_offset(
        std::size_t layer,
        BlockLocation location) const;
    void clear_slot(std::size_t layer, BlockLocation location);
    void clear_block(std::size_t physical_block);

    std::size_t layer_count_;
    std::size_t total_blocks_;
    std::size_t block_size_;
    std::size_t kv_heads_;
    std::size_t head_dim_;
    std::size_t payload_bytes_ = 0;
    KvBlockPool block_pool_;
    std::vector<float> keys_;
    std::vector<float> values_;
    std::vector<bool> written_;
};

}  // namespace chapter07_kv_cache
