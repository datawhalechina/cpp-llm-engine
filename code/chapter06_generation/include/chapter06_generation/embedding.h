// Chapter 06：声明 prefill 和 decode 使用的 token id -> embedding lookup。
// 契约：持有词表大小的 embedding table，返回确定性的 CPU 向量。
// 验证：generation 和 runtime smoke 测试检查 lookup 边界与输出 shape。
#pragma once

#include "chapter06_generation/generation_types.h"

#include <cstddef>

namespace chapter06_generation {

class Embedding {
public:
    Embedding(std::size_t vocab_size = 0);

    std::size_t vocab_size() const;
    bool ready() const;
    Logits forward(const TokenSequence& tokens) const;
    RaggedBatch forward_batch(const TokenBatch& token_rows) const;

private:
    std::size_t vocab_size_ = 0;
};

}  // namespace chapter06_generation
