// Chapter 07：实现请求页表和固定大小物理块池。
// 契约：预留失败不改变状态；只有已提交位置可被 locate 读取。
// 验证：block_pool_test.cpp 覆盖容量、提交、裁剪与复用。
#include "chapter07_kv_cache/block_table.h"

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace chapter07_kv_cache {

BlockTable::BlockTable(std::size_t block_size) : block_size_(block_size) {
    if (block_size_ == 0) throw std::invalid_argument("block size must be positive");
}

std::size_t BlockTable::block_size() const { return block_size_; }
std::size_t BlockTable::token_count() const { return token_count_; }
std::size_t BlockTable::block_count() const { return physical_blocks_.size(); }
std::size_t BlockTable::capacity_tokens() const {
    return physical_blocks_.size() * block_size_;
}
std::size_t BlockTable::immutable_prefix_tokens() const {
    return immutable_prefix_tokens_;
}
const std::vector<std::size_t>& BlockTable::physical_blocks() const {
    return physical_blocks_;
}

BlockLocation BlockTable::locate(std::size_t token_index) const {
    if (token_index >= token_count_) {
        throw std::out_of_range("token index exceeds committed block table");
    }
    return {physical_blocks_[token_index / block_size_], token_index % block_size_};
}

BlockLocation BlockTable::locate_reserved(std::size_t token_index) const {
    if (token_index >= capacity_tokens()) {
        throw std::out_of_range("token index exceeds reserved block capacity");
    }
    return {physical_blocks_[token_index / block_size_], token_index % block_size_};
}

void BlockTable::append_block(std::size_t physical_block) {
    physical_blocks_.push_back(physical_block);
}

void BlockTable::attach_prefix(
    const std::vector<std::size_t>& physical_blocks,
    std::size_t token_count) {
    if (!physical_blocks_.empty() || token_count == 0 ||
        token_count % block_size_ != 0 ||
        physical_blocks.size() != token_count / block_size_) {
        throw std::invalid_argument("shared prefix must contain complete blocks");
    }
    physical_blocks_ = physical_blocks;
    token_count_ = token_count;
    immutable_prefix_tokens_ = token_count;
}

std::size_t BlockTable::remove_last_block() {
    if (physical_blocks_.empty()) {
        throw std::logic_error("cannot remove a block from an empty table");
    }
    const std::size_t block = physical_blocks_.back();
    physical_blocks_.pop_back();
    return block;
}

void BlockTable::set_token_count(std::size_t token_count) {
    if (token_count > capacity_tokens()) {
        throw std::out_of_range("token count exceeds reserved block capacity");
    }
    token_count_ = token_count;
}

KvBlockPool::KvBlockPool(std::size_t total_blocks, std::size_t block_size)
    : block_size_(block_size) {
    if (total_blocks == 0 || block_size_ == 0) {
        throw std::invalid_argument("KV block count and block size must be positive");
    }
    if (total_blocks > std::numeric_limits<std::size_t>::max() / block_size_) {
        throw std::overflow_error("KV block capacity overflows size_t");
    }
    block_ref_counts_.assign(total_blocks, 0);
}

std::size_t KvBlockPool::total_blocks() const { return block_ref_counts_.size(); }
std::size_t KvBlockPool::free_blocks() const {
    return static_cast<std::size_t>(
        std::count(block_ref_counts_.begin(), block_ref_counts_.end(), 0));
}
std::size_t KvBlockPool::used_blocks() const { return total_blocks() - free_blocks(); }
std::size_t KvBlockPool::block_size() const { return block_size_; }
bool KvBlockPool::contains(std::size_t request_id) const {
    return tables_.find(request_id) != tables_.end();
}
std::size_t KvBlockPool::block_count(std::size_t request_id) const {
    const auto table = tables_.find(request_id);
    return table == tables_.end() ? 0 : table->second.block_count();
}
std::size_t KvBlockPool::token_count(std::size_t request_id) const {
    const auto table = tables_.find(request_id);
    return table == tables_.end() ? 0 : table->second.token_count();
}
std::size_t KvBlockPool::capacity_tokens(std::size_t request_id) const {
    const auto table = tables_.find(request_id);
    return table == tables_.end() ? 0 : table->second.capacity_tokens();
}
std::vector<std::size_t> KvBlockPool::physical_blocks(std::size_t request_id) const {
    const auto table = tables_.find(request_id);
    return table == tables_.end()
        ? std::vector<std::size_t>{}
        : table->second.physical_blocks();
}
std::size_t KvBlockPool::block_ref_count(std::size_t physical_block) const {
    return physical_block >= block_ref_counts_.size()
        ? 0 : block_ref_counts_[physical_block];
}

bool KvBlockPool::reserve(std::size_t request_id, std::size_t total_tokens) {
    const std::size_t target_blocks = required_blocks(total_tokens);
    auto table = tables_.find(request_id);
    const std::size_t current_blocks =
        table == tables_.end() ? 0 : table->second.block_count();
    if (target_blocks <= current_blocks) return true;
    const std::size_t additional_blocks = target_blocks - current_blocks;
    if (additional_blocks > free_blocks()) return false;

    if (table == tables_.end()) {
        table = tables_.emplace(request_id, BlockTable(block_size_)).first;
    }
    for (std::size_t block = 0, allocated = 0;
         block < block_ref_counts_.size() && allocated < additional_blocks;
         ++block) {
        if (block_ref_counts_[block] != 0) continue;
        block_ref_counts_[block] = 1;
        table->second.append_block(block);
        ++allocated;
    }
    return true;
}

bool KvBlockPool::commit_tokens(std::size_t request_id, std::size_t total_tokens) {
    const auto table = tables_.find(request_id);
    if (table == tables_.end() || total_tokens < table->second.token_count() ||
        total_tokens > table->second.capacity_tokens()) {
        return false;
    }
    table->second.set_token_count(total_tokens);
    return true;
}

bool KvBlockPool::trim_to_tokens(std::size_t request_id, std::size_t total_tokens) {
    auto table = tables_.find(request_id);
    if (table == tables_.end()) return total_tokens == 0;
    if (total_tokens > table->second.token_count() ||
        (total_tokens != 0 &&
         total_tokens < table->second.immutable_prefix_tokens())) return false;

    const std::size_t target_blocks = required_blocks(total_tokens);
    while (table->second.block_count() > target_blocks) {
        release_block(table->second.remove_last_block());
    }
    table->second.set_token_count(total_tokens);
    if (target_blocks == 0) tables_.erase(table);
    return true;
}

bool KvBlockPool::release(std::size_t request_id) {
    auto table = tables_.find(request_id);
    if (table == tables_.end()) return false;
    for (const std::size_t block : table->second.physical_blocks()) release_block(block);
    tables_.erase(table);
    return true;
}

bool KvBlockPool::attach_shared_prefix(
    std::size_t request_id,
    const std::vector<std::size_t>& physical_blocks,
    std::size_t token_count) {
    if (contains(request_id) || token_count == 0 || token_count % block_size_ != 0 ||
        physical_blocks.size() != required_blocks(token_count)) return false;
    std::vector<bool> seen(total_blocks(), false);
    for (const std::size_t block : physical_blocks) {
        if (block >= total_blocks() || seen[block] || block_ref_counts_[block] == 0 ||
            block_ref_counts_[block] == std::numeric_limits<std::size_t>::max()) {
            return false;
        }
        seen[block] = true;
    }
    BlockTable table(block_size_);
    table.attach_prefix(physical_blocks, token_count);
    for (const std::size_t block : physical_blocks) ++block_ref_counts_[block];
    tables_.emplace(request_id, std::move(table));
    return true;
}

bool KvBlockPool::retain_blocks(
    const std::vector<std::size_t>& physical_blocks) {
    if (physical_blocks.empty()) return false;
    std::vector<bool> seen(total_blocks(), false);
    for (const std::size_t block : physical_blocks) {
        if (block >= total_blocks() || seen[block] || block_ref_counts_[block] == 0 ||
            block_ref_counts_[block] == std::numeric_limits<std::size_t>::max()) {
            return false;
        }
        seen[block] = true;
    }
    for (const std::size_t block : physical_blocks) ++block_ref_counts_[block];
    return true;
}

bool KvBlockPool::release_retained_blocks(
    const std::vector<std::size_t>& physical_blocks) {
    if (physical_blocks.empty()) return false;
    std::vector<bool> seen(total_blocks(), false);
    for (const std::size_t block : physical_blocks) {
        if (block >= total_blocks() || seen[block] || block_ref_counts_[block] == 0) {
            return false;
        }
        seen[block] = true;
    }
    for (const std::size_t block : physical_blocks) release_block(block);
    return true;
}

BlockLocation KvBlockPool::locate(
    std::size_t request_id,
    std::size_t token_index) const {
    const auto table = tables_.find(request_id);
    if (table == tables_.end()) throw std::out_of_range("request has no KV block table");
    return table->second.locate(token_index);
}

BlockLocation KvBlockPool::locate_reserved(
    std::size_t request_id,
    std::size_t token_index) const {
    const auto table = tables_.find(request_id);
    if (table == tables_.end()) throw std::out_of_range("request has no KV block table");
    return table->second.locate_reserved(token_index);
}

std::size_t KvBlockPool::required_blocks(std::size_t token_count) const {
    return token_count / block_size_ + (token_count % block_size_ == 0 ? 0 : 1);
}

void KvBlockPool::release_block(std::size_t physical_block) {
    if (physical_block >= block_ref_counts_.size() ||
        block_ref_counts_[physical_block] == 0) {
        throw std::logic_error("cannot release an unreferenced KV block");
    }
    --block_ref_counts_[physical_block];
}

}  // namespace chapter07_kv_cache
