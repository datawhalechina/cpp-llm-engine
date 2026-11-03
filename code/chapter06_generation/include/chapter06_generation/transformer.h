// Chapter 06：声明 CPU Generation Runtime 使用的教学 Transformer assembly。
// 职责：组合 embedding、position encoding、decoder layer 和 logits production。
// 验证：generation 测试检查 assembled model 的 prefill/decode shape flow。
#pragma once

#include "chapter06_generation/decoder_layer.h"
#include "chapter06_generation/embedding.h"
#include "chapter06_generation/position_encoding.h"
#include "chapter06_generation/request.h"

#include <cstddef>
#include <vector>

namespace chapter06_generation {

class Transformer {
public:
    Transformer(
        std::size_t layer_count = 0,
        Embedding embedding = {},
        PositionEncoding position_encoding = {});

    std::size_t layer_count() const;
    const Embedding& embedding() const;
    const PositionEncoding& position_encoding() const;
    bool ready() const;
    Logits prefill(const Request& request) const;
    Logits decode(const Request& request) const;
    LogitsBatch prefill_batch(const std::vector<const Request*>& requests) const;
    LogitsBatch decode_batch(const std::vector<const Request*>& requests) const;

private:
    Logits forward(const Request& request, std::size_t step) const;
    LogitsBatch forward_batch(
        const std::vector<const Request*>& requests,
        const std::vector<std::size_t>& steps) const;

    std::vector<DecoderLayer> layers_;
    Embedding embedding_;
    PositionEncoding position_encoding_;
};

}  // namespace chapter06_generation
