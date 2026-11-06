// Chapter 06：实现 Request 词元历史的追加和清空。
// 边界：该对象只用于观察生成进度，模型 K/V 的物理存储属于 Chapter 07。
#include "chapter06_generation/token_history.h"

namespace chapter06_generation {

void TokenHistory::clear() {
    tokens_.clear();
}

void TokenHistory::append(TokenId token) {
    tokens_.push_back(token);
}

std::size_t TokenHistory::size() const {
    return tokens_.size();
}

const TokenSequence& TokenHistory::tokens() const {
    return tokens_;
}

bool TokenHistory::empty() const {
    return tokens_.empty();
}

}  // namespace chapter06_generation
