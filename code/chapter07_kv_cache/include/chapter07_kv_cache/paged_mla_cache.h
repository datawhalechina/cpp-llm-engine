// Chapter 07：声明 MLA 压缩状态的 CPU 分页数据面。
// 职责：按 layer/block/slot 保存 latent K/V 与 RoPE key。
// 边界：复用 BlockTable 事务，不执行 MLA 投影或 Attention。
#pragma once

#include "chapter07_kv_cache/block_table.h"

#include <cstddef>
#include <string>
#include <utility>
#include <vector>

namespace chapter07_kv_cache {

class CpuPagedMlaCache final : public KvCacheTransaction {
public:
    CpuPagedMlaCache(
        std::size_t layer_count,
        std::size_t total_blocks,
        std::size_t block_size,
        std::size_t latent_dimension,
        std::size_t rope_dimension);

    std::size_t layer_count() const;
    std::size_t total_blocks() const override;
    std::size_t free_blocks() const override;
    std::size_t used_blocks() const override;
    std::size_t block_size() const override;
    std::size_t latent_dimension() const;
    std::size_t rope_dimension() const;
    std::size_t payload_bytes() const;

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
    bool write_token(
        std::size_t request_id,
        std::size_t layer,
        std::size_t token_index,
        const float* latent,
        std::size_t latent_count,
        const float* key_rope,
        std::size_t rope_count,
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

    const float* latent_data(
        std::size_t request_id,
        std::size_t layer,
        std::size_t token_index) const;
    const float* key_rope_data(
        std::size_t request_id,
        std::size_t layer,
        std::size_t token_index) const;
    bool gather_layer(
        std::size_t request_id,
        std::size_t layer,
        std::vector<float>* latent,
        std::vector<float>* key_rope,
        std::string* error = nullptr) const;

private:
    std::size_t latent_offset(
        std::size_t layer,
        BlockLocation location) const;
    std::size_t rope_offset(
        std::size_t layer,
        BlockLocation location) const;
    std::size_t written_offset(
        std::size_t layer,
        BlockLocation location) const;
    void clear_slot(std::size_t layer, BlockLocation location);
    void clear_block(std::size_t physical_block);

    std::size_t layer_count_;
    std::size_t total_blocks_;
    std::size_t block_size_;
    std::size_t latent_dimension_;
    std::size_t rope_dimension_;
    std::size_t payload_bytes_ = 0;
    KvBlockPool block_pool_;
    std::vector<float> latent_;
    std::vector<float> key_rope_;
    std::vector<bool> written_;
};

}  // namespace chapter07_kv_cache
