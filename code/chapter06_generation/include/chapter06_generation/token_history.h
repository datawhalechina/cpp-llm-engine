// Chapter 06：声明 Request 持有的词元历史。
// 职责：记录已经提交给请求的 prompt 和生成词元，不保存模型 Attention K/V。
// 验证：TokenHistory 测试检查追加顺序、清空和请求隔离。
#pragma once

#include "chapter06_generation/generation_types.h"

namespace chapter06_generation {

class TokenHistory {
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
