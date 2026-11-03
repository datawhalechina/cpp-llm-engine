// Chapter 06：声明由 Request 持有的教学 KV Cache。
// 契约：在引入 block allocation 前，支持 token history 的 append、clear 和 size 观察。
// 验证：KV Cache 测试检查生命周期和请求隔离。
#pragma once

#include "chapter06_generation/generation_types.h"

namespace chapter06_generation {

class KVCache {
public:
    void clear();
    void append(TokenId token);

    std::size_t size() const;
    const TokenSequence& tokens() const;
    bool empty() const;

private:
    TokenSequence tokens_;
};

}  // namespace chapter06_generation
