// Chapter 07：实现 MLA 压缩状态的 CPU 分页数据面。
// 契约：所有层写入后才提交，裁剪与释放在物理块复用前清除载荷。
// 验证：paged_mla_cache_test.cpp 覆盖跨块顺序与事务原子性。
#include "chapter07_kv_cache/paged_mla_cache.h"

#include <algorithm>
#include <initializer_list>
#include <limits>
#include <set>
#include <stdexcept>

namespace chapter07_kv_cache {
namespace {

std::size_t checked_product(
    std::initializer_list<std::size_t> factors,
    const char* message) {
    std::size_t result = 1;
    for (const std::size_t factor : factors) {
        if (factor == 0) throw std::invalid_argument(message);
        if (result > std::numeric_limits<std::size_t>::max() / factor) {
            throw std::overflow_error(message);
        }
        result *= factor;
    }
    return result;
}

std::size_t checked_sum(
    std::size_t left,
    std::size_t right,
    const char* message) {
    if (left > std::numeric_limits<std::size_t>::max() - right) {
        throw std::overflow_error(message);
    }
    return left + right;
}

}  // namespace

CpuPagedMlaCache::CpuPagedMlaCache(
    std::size_t layer_count,
    std::size_t total_blocks,
    std::size_t block_size,
    std::size_t latent_dimension,
    std::size_t rope_dimension)
    : layer_count_(layer_count),
      total_blocks_(total_blocks),
      block_size_(block_size),
      latent_dimension_(latent_dimension),
      rope_dimension_(rope_dimension),
      block_pool_(total_blocks, block_size) {
    const std::size_t slot_count = checked_product(
        {layer_count_, total_blocks_, block_size_},
        "paged MLA slot dimensions must be positive and fit host size");
    const std::size_t latent_elements = checked_product(
        {slot_count, latent_dimension_},
        "paged MLA latent dimensions must fit host size");
    const std::size_t rope_elements = checked_product(
        {slot_count, rope_dimension_},
        "paged MLA RoPE dimensions must fit host size");
    const std::size_t payload_elements = checked_sum(
        latent_elements,
        rope_elements,
        "paged MLA payload dimensions must fit host size");
    payload_bytes_ = checked_product(
        {payload_elements, sizeof(float)},
        "paged MLA payload byte size must fit host size");
    latent_.assign(latent_elements, 0.0f);
    key_rope_.assign(rope_elements, 0.0f);
    written_.assign(slot_count, false);
}

std::size_t CpuPagedMlaCache::layer_count() const { return layer_count_; }
std::size_t CpuPagedMlaCache::total_blocks() const { return total_blocks_; }
std::size_t CpuPagedMlaCache::free_blocks() const {
    return block_pool_.free_blocks();
}
std::size_t CpuPagedMlaCache::used_blocks() const {
    return block_pool_.used_blocks();
}
std::size_t CpuPagedMlaCache::block_size() const { return block_size_; }
std::size_t CpuPagedMlaCache::latent_dimension() const {
    return latent_dimension_;
}
std::size_t CpuPagedMlaCache::rope_dimension() const {
    return rope_dimension_;
}
std::size_t CpuPagedMlaCache::payload_bytes() const { return payload_bytes_; }
bool CpuPagedMlaCache::contains(std::size_t request_id) const {
    return block_pool_.contains(request_id);
}
std::size_t CpuPagedMlaCache::block_count(std::size_t request_id) const {
    return block_pool_.block_count(request_id);
}
std::size_t CpuPagedMlaCache::token_count(std::size_t request_id) const {
    return block_pool_.token_count(request_id);
}
std::size_t CpuPagedMlaCache::capacity_tokens(std::size_t request_id) const {
    return block_pool_.capacity_tokens(request_id);
}
std::vector<std::size_t> CpuPagedMlaCache::physical_blocks(
    std::size_t request_id) const {
    return block_pool_.physical_blocks(request_id);
}
BlockLocation CpuPagedMlaCache::locate(
    std::size_t request_id,
    std::size_t token_index) const {
    return block_pool_.locate(request_id, token_index);
}
BlockLocation CpuPagedMlaCache::locate_reserved(
    std::size_t request_id,
    std::size_t token_index) const {
    return block_pool_.locate_reserved(request_id, token_index);
}

bool CpuPagedMlaCache::reserve(
    std::size_t request_id,
    std::size_t total_tokens) {
    return block_pool_.reserve(request_id, total_tokens);
}

bool CpuPagedMlaCache::write_token(
    std::size_t request_id,
    std::size_t layer,
    std::size_t token_index,
    const float* latent,
    std::size_t latent_count,
    const float* key_rope,
    std::size_t rope_count,
    std::string* error) {
    const auto fail = [&](const char* message) {
        if (error != nullptr) *error = message;
        return false;
    };
    if (layer >= layer_count_) return fail("paged MLA layer is out of range");
    if (latent == nullptr || key_rope == nullptr) {
        return fail("paged MLA input must not be null");
    }
    if (latent_count != latent_dimension_ || rope_count != rope_dimension_) {
        return fail("paged MLA token shape does not match");
    }
    if (!block_pool_.contains(request_id) ||
        token_index >= block_pool_.capacity_tokens(request_id)) {
        return fail("paged MLA token has no reserved slot");
    }
    if (token_index < block_pool_.token_count(request_id)) {
        return fail("paged MLA committed token is immutable");
    }

    const BlockLocation location =
        block_pool_.locate_reserved(request_id, token_index);
    std::copy_n(
        latent,
        latent_dimension_,
        latent_.begin() + latent_offset(layer, location));
    std::copy_n(
        key_rope,
        rope_dimension_,
        key_rope_.begin() + rope_offset(layer, location));
    written_[written_offset(layer, location)] = true;
    if (error != nullptr) error->clear();
    return true;
}

bool CpuPagedMlaCache::commit_tokens(
    std::size_t request_id,
    std::size_t total_tokens) {
    return commit_tokens(request_id, total_tokens, nullptr);
}

bool CpuPagedMlaCache::commit_tokens(
    std::size_t request_id,
    std::size_t total_tokens,
    std::string* error) {
    const auto fail = [&](const char* message) {
        if (error != nullptr) *error = message;
        return false;
    };
    if (!block_pool_.contains(request_id)) {
        return fail("request has no paged MLA table");
    }
    const std::size_t current_tokens = block_pool_.token_count(request_id);
    if (total_tokens < current_tokens ||
        total_tokens > block_pool_.capacity_tokens(request_id)) {
        return fail("paged MLA commit exceeds reserved capacity");
    }
    for (std::size_t token = current_tokens; token < total_tokens; ++token) {
        const BlockLocation location =
            block_pool_.locate_reserved(request_id, token);
        for (std::size_t layer = 0; layer < layer_count_; ++layer) {
            if (!written_[written_offset(layer, location)]) {
                return fail("paged MLA commit requires every layer to be written");
            }
        }
    }
    if (!block_pool_.commit_tokens(request_id, total_tokens)) {
        return fail("paged MLA block table rejected token commit");
    }
    if (error != nullptr) error->clear();
    return true;
}

bool CpuPagedMlaCache::commit_batch(
    const std::vector<std::pair<std::size_t, std::size_t>>& requests,
    std::string* error) {
    const auto fail = [&](const char* message) {
        if (error != nullptr) *error = message;
        return false;
    };
    if (requests.empty()) return fail("paged MLA batch commit requires requests");
    std::set<std::size_t> request_ids;
    for (const auto& [request_id, total_tokens] : requests) {
        if (!request_ids.insert(request_id).second) {
            return fail("paged MLA batch commit contains duplicate request");
        }
        if (!block_pool_.contains(request_id)) {
            return fail("paged MLA batch request has no table");
        }
        const std::size_t current_tokens = block_pool_.token_count(request_id);
        if (total_tokens < current_tokens ||
            total_tokens > block_pool_.capacity_tokens(request_id)) {
            return fail("paged MLA batch commit exceeds reserved capacity");
        }
        for (std::size_t token = current_tokens; token < total_tokens; ++token) {
            const BlockLocation location =
                block_pool_.locate_reserved(request_id, token);
            for (std::size_t layer = 0; layer < layer_count_; ++layer) {
                if (!written_[written_offset(layer, location)]) {
                    return fail("paged MLA batch commit requires every layer");
                }
            }
        }
    }
    for (const auto& [request_id, total_tokens] : requests) {
        if (!block_pool_.commit_tokens(request_id, total_tokens)) {
            throw std::logic_error("validated paged MLA batch commit failed");
        }
    }
    if (error != nullptr) error->clear();
    return true;
}

bool CpuPagedMlaCache::trim_to_tokens(
    std::size_t request_id,
    std::size_t total_tokens) {
    if (!block_pool_.contains(request_id)) return total_tokens == 0;
    if (total_tokens > block_pool_.token_count(request_id)) return false;
    const std::vector<std::size_t> blocks =
        block_pool_.physical_blocks(request_id);
    if (!block_pool_.trim_to_tokens(request_id, total_tokens)) return false;

    const std::size_t kept_blocks =
        total_tokens / block_size_ + (total_tokens % block_size_ == 0 ? 0 : 1);
    for (std::size_t index = kept_blocks; index < blocks.size(); ++index) {
        if (block_pool_.block_ref_count(blocks[index]) == 0) {
            clear_block(blocks[index]);
        }
    }
    if (total_tokens % block_size_ != 0 && kept_blocks != 0) {
        const std::size_t block = blocks[kept_blocks - 1];
        if (block_pool_.block_ref_count(block) == 1) {
            for (std::size_t slot = total_tokens % block_size_;
                 slot < block_size_; ++slot) {
                for (std::size_t layer = 0; layer < layer_count_; ++layer) {
                    clear_slot(layer, {block, slot});
                }
            }
        }
    }
    return true;
}

bool CpuPagedMlaCache::release(std::size_t request_id) {
    const std::vector<std::size_t> blocks =
        block_pool_.physical_blocks(request_id);
    if (blocks.empty()) return false;
    if (!block_pool_.release(request_id)) {
        throw std::logic_error("validated paged MLA release failed");
    }
    for (const std::size_t block : blocks) {
        if (block_pool_.block_ref_count(block) == 0) clear_block(block);
    }
    return true;
}

const float* CpuPagedMlaCache::latent_data(
    std::size_t request_id,
    std::size_t layer,
    std::size_t token_index) const {
    if (layer >= layer_count_) {
        throw std::out_of_range("paged MLA latent layer is out of range");
    }
    return latent_.data() + latent_offset(
        layer, block_pool_.locate(request_id, token_index));
}

const float* CpuPagedMlaCache::key_rope_data(
    std::size_t request_id,
    std::size_t layer,
    std::size_t token_index) const {
    if (layer >= layer_count_) {
        throw std::out_of_range("paged MLA RoPE layer is out of range");
    }
    return key_rope_.data() + rope_offset(
        layer, block_pool_.locate(request_id, token_index));
}

bool CpuPagedMlaCache::gather_layer(
    std::size_t request_id,
    std::size_t layer,
    std::vector<float>* latent,
    std::vector<float>* key_rope,
    std::string* error) const {
    const auto fail = [&](const char* message) {
        if (error != nullptr) *error = message;
        return false;
    };
    if (latent == nullptr || key_rope == nullptr) {
        return fail("paged MLA gather output must not be null");
    }
    if (layer >= layer_count_) return fail("paged MLA gather layer is out of range");
    if (!block_pool_.contains(request_id)) {
        return fail("request has no paged MLA table");
    }

    const std::size_t tokens = block_pool_.token_count(request_id);
    std::vector<float> gathered_latent(tokens * latent_dimension_);
    std::vector<float> gathered_rope(tokens * rope_dimension_);
    for (std::size_t token = 0; token < tokens; ++token) {
        const float* source_latent = latent_data(request_id, layer, token);
        const float* source_rope = key_rope_data(request_id, layer, token);
        std::copy_n(
            source_latent,
            latent_dimension_,
            gathered_latent.begin() + token * latent_dimension_);
        std::copy_n(
            source_rope,
            rope_dimension_,
            gathered_rope.begin() + token * rope_dimension_);
    }
    *latent = std::move(gathered_latent);
    *key_rope = std::move(gathered_rope);
    if (error != nullptr) error->clear();
    return true;
}

std::size_t CpuPagedMlaCache::latent_offset(
    std::size_t layer,
    BlockLocation location) const {
    return ((layer * total_blocks_ + location.block) * block_size_ +
            location.slot) * latent_dimension_;
}

std::size_t CpuPagedMlaCache::rope_offset(
    std::size_t layer,
    BlockLocation location) const {
    return ((layer * total_blocks_ + location.block) * block_size_ +
            location.slot) * rope_dimension_;
}

std::size_t CpuPagedMlaCache::written_offset(
    std::size_t layer,
    BlockLocation location) const {
    return (layer * total_blocks_ + location.block) * block_size_ +
        location.slot;
}

void CpuPagedMlaCache::clear_slot(
    std::size_t layer,
    BlockLocation location) {
    std::fill_n(
        latent_.begin() + latent_offset(layer, location),
        latent_dimension_,
        0.0f);
    std::fill_n(
        key_rope_.begin() + rope_offset(layer, location),
        rope_dimension_,
        0.0f);
    written_[written_offset(layer, location)] = false;
}

void CpuPagedMlaCache::clear_block(std::size_t physical_block) {
    for (std::size_t layer = 0; layer < layer_count_; ++layer) {
        for (std::size_t slot = 0; slot < block_size_; ++slot) {
            clear_slot(layer, {physical_block, slot});
        }
    }
}

}  // namespace chapter07_kv_cache
