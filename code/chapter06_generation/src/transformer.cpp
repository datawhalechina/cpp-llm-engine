// Chapter 06：实现 CPU Generation Runtime 使用的教学 multi-layer Transformer。
// 职责：将 token representation 经过 position encoding 和 decoder layer 送到 logits。
// 验证：generation 测试检查 assembled prefill 和 decode path。
#include "chapter06_generation/transformer.h"

#include <utility>

namespace chapter06_generation {

Transformer::Transformer(
    std::size_t layer_count,
    Embedding embedding,
    PositionEncoding position_encoding)
    : layers_(),
      embedding_(std::move(embedding)),
      position_encoding_(std::move(position_encoding)) {
    layers_.reserve(layer_count);
    for (std::size_t i = 0; i < layer_count; ++i) {
        layers_.emplace_back(i);
    }
}

std::size_t Transformer::layer_count() const {
    return layers_.size();
}

const Embedding& Transformer::embedding() const {
    return embedding_;
}

const PositionEncoding& Transformer::position_encoding() const {
    return position_encoding_;
}

bool Transformer::ready() const {
    return !layers_.empty() && embedding_.ready() && position_encoding_.ready();
}

Logits Transformer::prefill(const Request& request) const {
    const LogitsBatch batch = prefill_batch({&request});
    return batch.empty() ? Logits{} : batch.front();
}

Logits Transformer::decode(const Request& request) const {
    const LogitsBatch batch = decode_batch({&request});
    return batch.empty() ? Logits{} : batch.front();
}

Logits Transformer::forward(const Request& request, std::size_t step) const {
    const LogitsBatch batch = forward_batch({&request}, {step});
    return batch.empty() ? Logits{} : batch.front();
}

LogitsBatch Transformer::prefill_batch(const std::vector<const Request*>& requests) const {
    return forward_batch(requests, std::vector<std::size_t>(requests.size(), 0));
}

LogitsBatch Transformer::decode_batch(const std::vector<const Request*>& requests) const {
    std::vector<std::size_t> steps;
    steps.reserve(requests.size());
    for (const Request* request : requests) {
        if (request == nullptr) {
            return {};
        }
        steps.push_back(request->generated_size());
    }
    return forward_batch(requests, steps);
}

LogitsBatch Transformer::forward_batch(
    const std::vector<const Request*>& requests,
    const std::vector<std::size_t>& steps) const {
    if (!ready() || requests.empty() || requests.size() != steps.size()) {
        return {};
    }

    TokenBatch token_rows;
    token_rows.reserve(requests.size());
    for (const Request* request : requests) {
        if (request == nullptr) {
            return {};
        }
        TokenSequence tokens = request->prompt();
        tokens.insert(tokens.end(), request->generated().begin(), request->generated().end());
        token_rows.push_back(std::move(tokens));
    }

    RaggedBatch hidden = embedding_.forward_batch(token_rows);
    if (!position_encoding_.apply_batch(hidden, steps)) {
        return {};
    }
    for (const DecoderLayer& layer : layers_) {
        hidden = layer.forward_batch(hidden);
    }

    LogitsBatch outputs;
    outputs.reserve(requests.size());
    for (std::size_t row = 0; row < requests.size(); ++row) {
        const Request& request = *requests[row];
        const std::size_t begin = hidden.row_offsets[row];
        const std::size_t end = hidden.row_offsets[row + 1];
        const float base = begin == end ? 0.0f : hidden.values[begin];
        outputs.push_back({
            base + static_cast<float>(request.prompt_size()),
            base + static_cast<float>(request.generated_size()) + 1.0f,
            base + static_cast<float>(layers_.size()) + 2.0f,
        });
    }
    return outputs;
}

}  // namespace chapter06_generation
