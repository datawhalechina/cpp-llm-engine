// Chapter 06：实现 token ID 到 model vector 的确定性 CPU embedding lookup。
// 职责：提供 prefill 和 decode 消费的输入表示。
// 验证：Runtime 测试通过 assembled Transformer 覆盖 lookup path。
#include "chapter06_generation/embedding.h"

namespace chapter06_generation {

Embedding::Embedding(std::size_t vocab_size)
    : vocab_size_(vocab_size) {}

std::size_t Embedding::vocab_size() const {
    return vocab_size_;
}

bool Embedding::ready() const {
    return vocab_size_ > 0;
}

Logits Embedding::forward(const TokenSequence& tokens) const {
    Logits hidden;
    hidden.reserve(tokens.size());
    for (TokenId token : tokens) {
        const float normalized = vocab_size_ > 0 ? static_cast<float>(token % static_cast<TokenId>(vocab_size_)) : static_cast<float>(token);
        hidden.push_back(normalized);
    }
    return hidden;
}

RaggedBatch Embedding::forward_batch(const TokenBatch& token_rows) const {
    RaggedBatch hidden;
    hidden.row_offsets.reserve(token_rows.size() + 1);
    hidden.row_offsets.push_back(0);
    for (const TokenSequence& tokens : token_rows) {
        const Logits row = forward(tokens);
        hidden.values.insert(hidden.values.end(), row.begin(), row.end());
        hidden.row_offsets.push_back(hidden.values.size());
    }
    return hidden;
}

}  // namespace chapter06_generation
