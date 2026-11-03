// Chapter 06：声明 CPU 分页 K/V 数据面。
// 职责：组合 BlockTable 控制面，并按 layer/block/slot/head/dim 保存真实 K/V 数值。
// 验证：paged_kv_cache_test.cpp 检查跨块读写、Attention 等价和释放复用隔离。
#pragma once

#include "chapter06_generation/serving_preview.h"

#include <cstddef>
#include <string>
#include <vector>

namespace chapter06_generation {

class CpuPagedKvCache final {
public:
    CpuPagedKvCache(
        std::size_t layer_count,
        std::size_t total_blocks,
        std::size_t block_size,
        std::size_t kv_heads,
        std::size_t head_dim);

    std::size_t layer_count() const;
    std::size_t total_blocks() const;
    std::size_t block_size() const;
    std::size_t kv_heads() const;
    std::size_t head_dim() const;
    std::size_t payload_bytes() const;

    const KvBlockPool& block_pool() const;
    bool reserve(std::size_t request_id, std::size_t total_tokens);
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
        std::size_t total_tokens,
        std::string* error = nullptr);
    bool trim_to_tokens(std::size_t request_id, std::size_t total_tokens);
    bool release(std::size_t request_id);

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
    KvBlockPool block_pool_;
    std::vector<float> keys_;
    std::vector<float> values_;
    std::vector<bool> written_;
};

}  // namespace chapter06_generation
