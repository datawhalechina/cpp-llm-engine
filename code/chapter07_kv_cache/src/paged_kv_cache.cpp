// Chapter 07：实现 CPU 分页 K/V 数据面和单词元 GQA 参考路径。
// 契约：所有层均写入后才能提交，读取仅访问已提交位置。
// 验证：paged_kv_cache_test.cpp 对照连续 Attention 并检查块复用隔离。
#include "chapter07_kv_cache/paged_kv_cache.h"

#include <algorithm>
#include <cmath>
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

float dot(const float* left, const float* right, std::size_t size) {
    float result = 0.0f;
    for (std::size_t index = 0; index < size; ++index) {
        result += left[index] * right[index];
    }
    return result;
}

}  // namespace

CpuPagedKvCache::CpuPagedKvCache(
    std::size_t layer_count,
    std::size_t total_blocks,
    std::size_t block_size,
    std::size_t kv_heads,
    std::size_t head_dim)
    : layer_count_(layer_count),
      total_blocks_(total_blocks),
      block_size_(block_size),
      kv_heads_(kv_heads),
      head_dim_(head_dim),
      block_pool_(total_blocks, block_size) {
    const std::size_t slot_count = checked_product(
        {layer_count_, total_blocks_, block_size_},
        "paged KV slot dimensions must be positive and fit host size");
    const std::size_t element_count = checked_product(
        {slot_count, kv_heads_, head_dim_},
        "paged KV payload dimensions must fit host size");
    payload_bytes_ = checked_product(
        {element_count, 2, sizeof(float)},
        "paged KV payload byte size must fit host size");
    keys_.assign(element_count, 0.0f);
    values_.assign(element_count, 0.0f);
    written_.assign(slot_count, false);
}

std::size_t CpuPagedKvCache::layer_count() const { return layer_count_; }
std::size_t CpuPagedKvCache::total_blocks() const { return total_blocks_; }
std::size_t CpuPagedKvCache::free_blocks() const { return block_pool_.free_blocks(); }
std::size_t CpuPagedKvCache::used_blocks() const { return block_pool_.used_blocks(); }
std::size_t CpuPagedKvCache::block_size() const { return block_size_; }
std::size_t CpuPagedKvCache::kv_heads() const { return kv_heads_; }
std::size_t CpuPagedKvCache::head_dim() const { return head_dim_; }
std::size_t CpuPagedKvCache::payload_bytes() const { return payload_bytes_; }
const KvBlockPool& CpuPagedKvCache::block_pool() const { return block_pool_; }
bool CpuPagedKvCache::contains(std::size_t request_id) const {
    return block_pool_.contains(request_id);
}
std::size_t CpuPagedKvCache::block_count(std::size_t request_id) const {
    return block_pool_.block_count(request_id);
}
std::size_t CpuPagedKvCache::token_count(std::size_t request_id) const {
    return block_pool_.token_count(request_id);
}
std::size_t CpuPagedKvCache::capacity_tokens(std::size_t request_id) const {
    return block_pool_.capacity_tokens(request_id);
}
std::vector<std::size_t> CpuPagedKvCache::physical_blocks(
    std::size_t request_id) const {
    return block_pool_.physical_blocks(request_id);
}
BlockLocation CpuPagedKvCache::locate(
    std::size_t request_id,
    std::size_t token_index) const {
    return block_pool_.locate(request_id, token_index);
}
BlockLocation CpuPagedKvCache::locate_reserved(
    std::size_t request_id,
    std::size_t token_index) const {
    return block_pool_.locate_reserved(request_id, token_index);
}

bool CpuPagedKvCache::reserve(std::size_t request_id, std::size_t total_tokens) {
    return block_pool_.reserve(request_id, total_tokens);
}

bool CpuPagedKvCache::attach_shared_prefix(
    std::size_t request_id,
    const std::vector<std::size_t>& physical_blocks,
    std::size_t token_count) {
    return block_pool_.attach_shared_prefix(
        request_id, physical_blocks, token_count);
}

bool CpuPagedKvCache::retain_prefix_blocks(
    const std::vector<std::size_t>& physical_blocks) {
    return block_pool_.retain_blocks(physical_blocks);
}

bool CpuPagedKvCache::release_prefix_blocks(
    const std::vector<std::size_t>& physical_blocks) {
    if (!block_pool_.release_retained_blocks(physical_blocks)) return false;
    for (const std::size_t block : physical_blocks) {
        if (block_pool_.block_ref_count(block) == 0) clear_block(block);
    }
    return true;
}

std::size_t CpuPagedKvCache::block_ref_count(
    std::size_t physical_block) const {
    return block_pool_.block_ref_count(physical_block);
}

bool CpuPagedKvCache::write_token(
    std::size_t request_id,
    std::size_t layer,
    std::size_t token_index,
    const float* key,
    const float* value,
    std::size_t element_count,
    std::string* error) {
    const auto fail = [&](const char* message) {
        if (error != nullptr) *error = message;
        return false;
    };
    if (layer >= layer_count_) return fail("paged KV layer is out of range");
    if (key == nullptr || value == nullptr) return fail("paged KV input must not be null");
    const std::size_t token_width = kv_heads_ * head_dim_;
    if (element_count != token_width) return fail("paged KV token width does not match");
    if (!block_pool_.contains(request_id) ||
        token_index >= block_pool_.capacity_tokens(request_id)) {
        return fail("paged KV token has no reserved slot");
    }
    if (token_index < block_pool_.token_count(request_id)) {
        return fail("paged KV committed token is immutable");
    }

    const BlockLocation location = block_pool_.locate_reserved(request_id, token_index);
    const std::size_t offset = payload_offset(layer, location, 0);
    std::copy_n(key, token_width, keys_.begin() + offset);
    std::copy_n(value, token_width, values_.begin() + offset);
    written_[written_offset(layer, location)] = true;
    if (error != nullptr) error->clear();
    return true;
}

bool CpuPagedKvCache::commit_tokens(
    std::size_t request_id,
    std::size_t total_tokens) {
    return commit_tokens(request_id, total_tokens, nullptr);
}

bool CpuPagedKvCache::commit_tokens(
    std::size_t request_id,
    std::size_t total_tokens,
    std::string* error) {
    const auto fail = [&](const char* message) {
        if (error != nullptr) *error = message;
        return false;
    };
    if (!block_pool_.contains(request_id)) return fail("request has no paged KV table");
    const std::size_t current_tokens = block_pool_.token_count(request_id);
    if (total_tokens < current_tokens ||
        total_tokens > block_pool_.capacity_tokens(request_id)) {
        return fail("paged KV commit exceeds reserved capacity");
    }
    for (std::size_t token = current_tokens; token < total_tokens; ++token) {
        const BlockLocation location = block_pool_.locate_reserved(request_id, token);
        for (std::size_t layer = 0; layer < layer_count_; ++layer) {
            if (!written_[written_offset(layer, location)]) {
                return fail("paged KV commit requires every layer to be written");
            }
        }
    }
    if (!block_pool_.commit_tokens(request_id, total_tokens)) {
        return fail("paged KV block table rejected token commit");
    }
    if (error != nullptr) error->clear();
    return true;
}

bool CpuPagedKvCache::commit_batch(
    const std::vector<std::pair<std::size_t, std::size_t>>& requests,
    std::string* error) {
    const auto fail = [&](const char* message) {
        if (error != nullptr) *error = message;
        return false;
    };
    if (requests.empty()) return fail("paged KV batch commit requires requests");
    std::set<std::size_t> request_ids;
    for (const auto& [request_id, total_tokens] : requests) {
        if (!request_ids.insert(request_id).second) {
            return fail("paged KV batch commit contains duplicate request");
        }
        if (!block_pool_.contains(request_id)) {
            return fail("paged KV batch request has no table");
        }
        const std::size_t current_tokens = block_pool_.token_count(request_id);
        if (total_tokens < current_tokens ||
            total_tokens > block_pool_.capacity_tokens(request_id)) {
            return fail("paged KV batch commit exceeds reserved capacity");
        }
        for (std::size_t token = current_tokens; token < total_tokens; ++token) {
            const BlockLocation location =
                block_pool_.locate_reserved(request_id, token);
            for (std::size_t layer = 0; layer < layer_count_; ++layer) {
                if (!written_[written_offset(layer, location)]) {
                    return fail("paged KV batch commit requires every layer");
                }
            }
        }
    }
    for (const auto& [request_id, total_tokens] : requests) {
        if (!block_pool_.commit_tokens(request_id, total_tokens)) {
            throw std::logic_error("validated paged KV batch commit failed");
        }
    }
    if (error != nullptr) error->clear();
    return true;
}

bool CpuPagedKvCache::trim_to_tokens(
    std::size_t request_id,
    std::size_t total_tokens) {
    if (!block_pool_.contains(request_id)) return total_tokens == 0;
    if (total_tokens > block_pool_.token_count(request_id)) return false;
    const std::vector<std::size_t> blocks = block_pool_.physical_blocks(request_id);
    if (!block_pool_.trim_to_tokens(request_id, total_tokens)) return false;

    const std::size_t kept_blocks =
        total_tokens / block_size_ + (total_tokens % block_size_ == 0 ? 0 : 1);
    for (std::size_t index = kept_blocks; index < blocks.size(); ++index) {
        if (block_pool_.block_ref_count(blocks[index]) == 0) clear_block(blocks[index]);
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

bool CpuPagedKvCache::release(std::size_t request_id) {
    const std::vector<std::size_t> blocks = block_pool_.physical_blocks(request_id);
    if (blocks.empty()) return false;
    if (!block_pool_.release(request_id)) {
        throw std::logic_error("validated paged KV release failed");
    }
    for (const std::size_t block : blocks) {
        if (block_pool_.block_ref_count(block) == 0) clear_block(block);
    }
    return true;
}

const float* CpuPagedKvCache::key_data(
    std::size_t request_id,
    std::size_t layer,
    std::size_t token_index,
    std::size_t kv_head) const {
    if (layer >= layer_count_ || kv_head >= kv_heads_) {
        throw std::out_of_range("paged KV key index is out of range");
    }
    return keys_.data() + payload_offset(
        layer, block_pool_.locate(request_id, token_index), kv_head);
}

const float* CpuPagedKvCache::value_data(
    std::size_t request_id,
    std::size_t layer,
    std::size_t token_index,
    std::size_t kv_head) const {
    if (layer >= layer_count_ || kv_head >= kv_heads_) {
        throw std::out_of_range("paged KV value index is out of range");
    }
    return values_.data() + payload_offset(
        layer, block_pool_.locate(request_id, token_index), kv_head);
}

const float* CpuPagedKvCache::key_data_reserved(
    std::size_t request_id,
    std::size_t layer,
    std::size_t token_index,
    std::size_t kv_head) const {
    if (layer >= layer_count_ || kv_head >= kv_heads_) {
        throw std::out_of_range("paged KV reserved key index is out of range");
    }
    const BlockLocation location = block_pool_.locate_reserved(request_id, token_index);
    if (!written_[written_offset(layer, location)]) {
        throw std::logic_error("paged KV reserved key has not been written");
    }
    return keys_.data() + payload_offset(layer, location, kv_head);
}

const float* CpuPagedKvCache::value_data_reserved(
    std::size_t request_id,
    std::size_t layer,
    std::size_t token_index,
    std::size_t kv_head) const {
    if (layer >= layer_count_ || kv_head >= kv_heads_) {
        throw std::out_of_range("paged KV reserved value index is out of range");
    }
    const BlockLocation location = block_pool_.locate_reserved(request_id, token_index);
    if (!written_[written_offset(layer, location)]) {
        throw std::logic_error("paged KV reserved value has not been written");
    }
    return values_.data() + payload_offset(layer, location, kv_head);
}

bool CpuPagedKvCache::attention(
    std::size_t request_id,
    std::size_t layer,
    const std::vector<float>& query,
    std::size_t query_heads,
    std::vector<float>* output,
    std::string* error) const {
    const auto fail = [&](const char* message) {
        if (error != nullptr) *error = message;
        return false;
    };
    if (output == nullptr) return fail("paged attention output must not be null");
    if (layer >= layer_count_) return fail("paged attention layer is out of range");
    if (query_heads > std::numeric_limits<std::size_t>::max() / head_dim_) {
        return fail("paged attention query shape overflows host size");
    }
    const std::size_t query_width = query_heads * head_dim_;
    if (query_heads == 0 || query_heads % kv_heads_ != 0 ||
        query.size() != query_width) {
        return fail("paged attention query shape does not match GQA layout");
    }
    const std::size_t token_count = block_pool_.token_count(request_id);
    if (token_count == 0) return fail("paged attention requires committed K/V tokens");

    std::vector<float> result(query_width, 0.0f);
    const std::size_t queries_per_kv_head = query_heads / kv_heads_;
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim_));
    std::vector<float> scores(token_count, 0.0f);
    for (std::size_t query_head = 0; query_head < query_heads; ++query_head) {
        const std::size_t kv_head = query_head / queries_per_kv_head;
        const float* query_row = query.data() + query_head * head_dim_;
        for (std::size_t token = 0; token < token_count; ++token) {
            scores[token] = dot(
                query_row, key_data(request_id, layer, token, kv_head), head_dim_) * scale;
        }
        const float maximum = *std::max_element(scores.begin(), scores.end());
        float denominator = 0.0f;
        for (float& score : scores) {
            score = std::exp(score - maximum);
            denominator += score;
        }
        if (!std::isfinite(denominator) || denominator <= 0.0f) {
            return fail("paged attention softmax denominator is invalid");
        }
        float* output_row = result.data() + query_head * head_dim_;
        for (std::size_t token = 0; token < token_count; ++token) {
            const float probability = scores[token] / denominator;
            const float* value = value_data(request_id, layer, token, kv_head);
            for (std::size_t dimension = 0; dimension < head_dim_; ++dimension) {
                output_row[dimension] += probability * value[dimension];
            }
        }
    }
    *output = std::move(result);
    if (error != nullptr) error->clear();
    return true;
}

std::size_t CpuPagedKvCache::payload_offset(
    std::size_t layer,
    BlockLocation location,
    std::size_t kv_head) const {
    return ((((layer * total_blocks_ + location.block) * block_size_ + location.slot) *
             kv_heads_ + kv_head) * head_dim_);
}

std::size_t CpuPagedKvCache::written_offset(
    std::size_t layer,
    BlockLocation location) const {
    return (layer * total_blocks_ + location.block) * block_size_ + location.slot;
}

void CpuPagedKvCache::clear_slot(std::size_t layer, BlockLocation location) {
    const std::size_t offset = payload_offset(layer, location, 0);
    const std::size_t width = kv_heads_ * head_dim_;
    std::fill_n(keys_.begin() + offset, width, 0.0f);
    std::fill_n(values_.begin() + offset, width, 0.0f);
    written_[written_offset(layer, location)] = false;
}

void CpuPagedKvCache::clear_block(std::size_t physical_block) {
    for (std::size_t layer = 0; layer < layer_count_; ++layer) {
        for (std::size_t slot = 0; slot < block_size_; ++slot) {
            clear_slot(layer, {physical_block, slot});
        }
    }
}

}  // namespace chapter07_kv_cache
