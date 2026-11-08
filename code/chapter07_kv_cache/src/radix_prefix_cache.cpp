// Chapter 07：实现块对齐的 CPU Radix 前缀缓存。
// 契约：索引和请求分别持有块引用，最长命中只返回完整物理块。
// 验证：radix_prefix_cache_test.cpp 覆盖命中、共享、释放和私有后缀。
#include "chapter07_kv_cache/radix_prefix_cache.h"

#include <stdexcept>

namespace chapter07_kv_cache {

RadixPrefixCache::RadixPrefixCache(CpuPagedKvCache* cache) : cache_(cache) {
    if (cache_ == nullptr) {
        throw std::invalid_argument("Radix prefix cache requires a KV cache");
    }
}

RadixPrefixCache::~RadixPrefixCache() { clear(); }

bool RadixPrefixCache::publish(
    const std::vector<int>& tokens,
    std::size_t request_id) {
    const std::size_t aligned_tokens =
        tokens.size() / cache_->block_size() * cache_->block_size();
    if (aligned_tokens == 0 ||
        cache_->token_count(request_id) < aligned_tokens) return false;

    const std::vector<std::size_t> blocks = cache_->physical_blocks(request_id);
    Node* node = &root_;
    for (std::size_t token = 0; token < aligned_tokens; ++token) {
        auto& child = node->children[tokens[token]];
        if (!child) child = std::make_unique<Node>();
        node = child.get();
        if ((token + 1) % cache_->block_size() != 0) continue;

        const std::size_t physical_block = blocks[token / cache_->block_size()];
        if (node->has_physical_block) continue;
        if (!cache_->retain_prefix_blocks({physical_block})) return false;
        node->has_physical_block = true;
        node->physical_block = physical_block;
        ++cached_blocks_;
    }
    return true;
}

PrefixMatch RadixPrefixCache::longest_prefix(
    const std::vector<int>& tokens) const {
    PrefixMatch match;
    const Node* node = &root_;
    for (std::size_t token = 0; token < tokens.size(); ++token) {
        const auto child = node->children.find(tokens[token]);
        if (child == node->children.end()) break;
        node = child->second.get();
        if ((token + 1) % cache_->block_size() == 0) {
            if (!node->has_physical_block) break;
            match.token_count = token + 1;
            match.physical_blocks.push_back(node->physical_block);
        }
    }
    return match;
}

bool RadixPrefixCache::attach(
    std::size_t request_id,
    const PrefixMatch& match) {
    if (match.token_count == 0 ||
        match.physical_blocks.size() * cache_->block_size() !=
            match.token_count) return false;
    return cache_->attach_shared_prefix(
        request_id, match.physical_blocks, match.token_count);
}

void RadixPrefixCache::clear() {
    release_node(&root_);
    root_.children.clear();
    cached_blocks_ = 0;
}

std::size_t RadixPrefixCache::cached_blocks() const { return cached_blocks_; }

void RadixPrefixCache::release_node(Node* node) {
    for (auto& [token, child] : node->children) {
        static_cast<void>(token);
        release_node(child.get());
    }
    if (node->has_physical_block) {
        if (!cache_->release_prefix_blocks({node->physical_block})) {
            throw std::logic_error("Radix prefix block ownership is inconsistent");
        }
        node->has_physical_block = false;
    }
}

}  // namespace chapter07_kv_cache
