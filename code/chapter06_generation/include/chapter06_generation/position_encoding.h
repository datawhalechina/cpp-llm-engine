// Chapter 06：声明教学 Transformer 使用的确定性 position encoding。
// 契约：把 token position 映射为向量，不改变 Runtime 的 Request contract。
// 验证：Runtime 测试检查 sequence length 和 decode step 的衔接。
#pragma once

#include "chapter06_generation/generation_types.h"

#include <cstddef>
#include <vector>

namespace chapter06_generation {

class PositionEncoding {
public:
    PositionEncoding(std::size_t max_length = 0);

    std::size_t max_length() const;
    bool ready() const;
    void apply(Logits& hidden, std::size_t position_base) const;
    bool apply_batch(RaggedBatch& hidden, const std::vector<std::size_t>& position_bases) const;

private:
    std::size_t max_length_ = 0;
};

}  // namespace chapter06_generation
