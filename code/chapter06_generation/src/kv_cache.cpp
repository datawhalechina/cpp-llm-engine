// Chapter 06：实现教学 Runtime 使用的 request-owned token-history cache。
// 边界：当前只表达 append 和 lifecycle ownership，不实现生产级 paged allocation。
// 验证：KV Cache 测试覆盖 append、clear 和独立 request state。
#include "chapter06_generation/kv_cache.h"

namespace chapter06_generation {

void KVCache::clear() {
    tokens_.clear();
}

void KVCache::append(TokenId token) {
    tokens_.push_back(token);
}

std::size_t KVCache::size() const {
    return tokens_.size();
}

const TokenSequence& KVCache::tokens() const {
    return tokens_;
}

bool KVCache::empty() const {
    return tokens_.empty();
}

}  // namespace chapter06_generation
