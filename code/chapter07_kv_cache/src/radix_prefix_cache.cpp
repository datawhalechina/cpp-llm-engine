// Chapter 07：实现块对齐的 CPU Radix 前缀缓存。
// 契约：索引和请求分别持有块引用，最长命中只返回完整物理块。
// 验证：radix_prefix_cache_test.cpp 覆盖命中、共享、统计、活跃保护和叶端淘汰。
#include "chapter07_kv_cache/radix_prefix_cache.h"

#include <limits>
#include <stdexcept>

namespace chapter07_kv_cache {

RadixPrefixCache::RadixPrefixCache(PrefixBlockStore* store) : store_(store) {
    if (store_ == nullptr) {
        throw std::invalid_argument("Radix prefix cache requires a KV cache");
    }
}

RadixPrefixCache::~RadixPrefixCache() { clear(); }

bool RadixPrefixCache::publish(
    const std::vector<int>& tokens,
    std::size_t request_id) {
    const std::size_t aligned_tokens =
        tokens.size() / store_->block_size() * store_->block_size();
    if (aligned_tokens == 0 ||
        store_->token_count(request_id) < aligned_tokens) return false;

    const std::vector<std::size_t> blocks = store_->physical_blocks(request_id);
    Node* node = &root_;
    for (std::size_t token = 0; token < aligned_tokens; ++token) {
        auto& child = node->children[tokens[token]];
        if (!child) child = std::make_unique<Node>();
        node = child.get();
        if ((token + 1) % store_->block_size() != 0) continue;

        const std::size_t physical_block = blocks[token / store_->block_size()];
        if (node->has_physical_block) {
            node->last_access = ++access_clock_;
            continue;
        }
        if (!store_->retain_prefix_blocks({physical_block})) return false;
        node->has_physical_block = true;
        node->physical_block = physical_block;
        node->last_access = ++access_clock_;
        ++cached_blocks_;
    }
    return true;
}

std::size_t RadixPrefixCache::estimate_prefix_tokens(
    const std::vector<int>& tokens) const {
    std::size_t matched_tokens = 0;
    const Node* node = &root_;
    for (std::size_t token = 0; token < tokens.size(); ++token) {
        const auto child = node->children.find(tokens[token]);
        if (child == node->children.end()) break;
        node = child->second.get();
        if ((token + 1) % store_->block_size() == 0) {
            if (!node->has_physical_block) break;
            matched_tokens = token + 1;
        }
    }
    return matched_tokens;
}

PrefixMatch RadixPrefixCache::longest_prefix(
    const std::vector<int>& tokens) const {
    ++stats_.queries;
    PrefixMatch match;
    const Node* node = &root_;
    for (std::size_t token = 0; token < tokens.size(); ++token) {
        const auto child = node->children.find(tokens[token]);
        if (child == node->children.end()) break;
        node = child->second.get();
        if ((token + 1) % store_->block_size() == 0) {
            if (!node->has_physical_block) break;
            match.token_count = token + 1;
            match.physical_blocks.push_back(node->physical_block);
            node->last_access = ++access_clock_;
        }
    }
    if (match.token_count == 0) {
        ++stats_.misses;
    } else {
        ++stats_.hits;
        stats_.matched_tokens += match.token_count;
    }
    return match;
}

bool RadixPrefixCache::attach(
    std::size_t request_id,
    const PrefixMatch& match) {
    if (match.token_count == 0 ||
        match.physical_blocks.size() * store_->block_size() !=
            match.token_count) return false;
    return store_->attach_shared_prefix(
        request_id, match.physical_blocks, match.token_count);
}

bool RadixPrefixCache::evict_until_free(std::size_t required_free_blocks) {
    if (required_free_blocks > store_->total_blocks()) return false;
    while (store_->free_blocks() < required_free_blocks) {
        if (!evict_one()) return false;
    }
    return true;
}

void RadixPrefixCache::clear() {
    release_node(&root_);
    root_.children.clear();
    cached_blocks_ = 0;
}

std::size_t RadixPrefixCache::cached_blocks() const { return cached_blocks_; }

RadixPrefixCacheStats RadixPrefixCache::stats() const { return stats_; }

bool RadixPrefixCache::find_eviction_candidate(
    Node* node,
    Node** candidate,
    std::size_t* oldest_access) const {
    bool has_cached_descendant = false;
    for (auto& [token, child] : node->children) {
        static_cast<void>(token);
        if (find_eviction_candidate(child.get(), candidate, oldest_access)) {
            has_cached_descendant = true;
        }
    }
    if (node->has_physical_block && !has_cached_descendant &&
        store_->block_ref_count(node->physical_block) == 1 &&
        node->last_access < *oldest_access) {
        *candidate = node;
        *oldest_access = node->last_access;
    }
    return node->has_physical_block || has_cached_descendant;
}

bool RadixPrefixCache::evict_one() {
    Node* candidate = nullptr;
    std::size_t oldest_access = std::numeric_limits<std::size_t>::max();
    find_eviction_candidate(&root_, &candidate, &oldest_access);
    if (candidate == nullptr ||
        !store_->release_prefix_blocks({candidate->physical_block})) {
        return false;
    }
    candidate->has_physical_block = false;
    candidate->physical_block = 0;
    candidate->last_access = 0;
    --cached_blocks_;
    ++stats_.evicted_blocks;
    return true;
}

void RadixPrefixCache::release_node(Node* node) {
    for (auto& [token, child] : node->children) {
        static_cast<void>(token);
        release_node(child.get());
    }
    if (node->has_physical_block) {
        if (!store_->release_prefix_blocks({node->physical_block})) {
            throw std::logic_error("Radix prefix block ownership is inconsistent");
        }
        node->has_physical_block = false;
    }
}

}  // namespace chapter07_kv_cache
