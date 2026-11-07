// Core：Decoder 结构与 token 执行接口
// 结构对象记录模型维度；执行上下文保存单个请求的 prefill/decode 状态。
// Tensor、权重和具体数学实现由 Chapter04、CPU 模型或 CUDA 后端提供。
// 验证：Core、Chapter04 与真实 TinyLlama Engine 测试。
#pragma once

#include "status.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace cpp_llm_engine::core {

enum class DecoderNormalizationKind : std::uint8_t {
    LayerNorm,
    RmsNorm,
};

enum class DecoderActivationKind : std::uint8_t {
    Relu,
    SwiGlu,
};

enum class DecoderPositionEncodingKind : std::uint8_t {
    None,
    Rotary,
};

struct DecoderLayerContract final {
    std::size_t hidden_size = 0;
    std::size_t ffn_hidden_size = 0;
    std::size_t query_head_count = 0;
    std::size_t kv_head_count = 0;
    std::size_t head_dimension = 0;
    std::size_t position_dimension = 0;
    DecoderNormalizationKind normalization = DecoderNormalizationKind::LayerNorm;
    DecoderActivationKind activation = DecoderActivationKind::Relu;
    DecoderPositionEncodingKind position_encoding = DecoderPositionEncodingKind::None;
    bool causal_attention = true;
    bool uses_kv_cache = false;

    Status validate() const {
        if (hidden_size == 0 || ffn_hidden_size == 0 || query_head_count == 0 ||
            kv_head_count == 0 || head_dimension == 0) {
            return Status::error("decoder dimensions and head counts must be non-zero");
        }
        if (query_head_count > std::numeric_limits<std::size_t>::max() / head_dimension ||
            query_head_count * head_dimension != hidden_size) {
            return Status::error("decoder query heads do not match hidden size");
        }
        if (query_head_count % kv_head_count != 0) {
            return Status::error("decoder query heads must be divisible by KV heads");
        }
        if (position_encoding == DecoderPositionEncodingKind::Rotary &&
            (position_dimension == 0 || position_dimension > head_dimension ||
             position_dimension % 2 != 0)) {
            return Status::error(
                "decoder rotary dimension must be even and fit inside one head");
        }
        return Status::ok();
    }
};

struct DecoderModelContract final {
    std::size_t vocab_size = 0;
    std::size_t layer_count = 0;
    std::size_t max_context_tokens = 0;
    DecoderLayerContract layer;

    Status validate() const {
        if (vocab_size == 0 || layer_count == 0 || max_context_tokens == 0) {
            return Status::error("decoder model sizes must be non-zero");
        }
        return layer.validate();
    }
};

using DecoderTokenId = std::int32_t;
using DecoderTokenSequence = std::vector<DecoderTokenId>;
using DecoderLogits = std::vector<float>;

struct DecoderEmbeddingSequence final {
    std::size_t token_count = 0;
    std::size_t hidden_size = 0;
    std::vector<float> values;

    Status validate(std::size_t expected_hidden_size) const {
        if (token_count == 0 || hidden_size == 0 || hidden_size != expected_hidden_size) {
            return Status::error("decoder embedding sequence has an invalid shape");
        }
        if (token_count > std::numeric_limits<std::size_t>::max() / hidden_size ||
            values.size() != token_count * hidden_size) {
            return Status::error("decoder embedding sequence storage does not match shape");
        }
        return Status::ok();
    }
};

class DecoderExecutionContext {
public:
    virtual ~DecoderExecutionContext() = default;
    virtual void reset() = 0;
    virtual Status prefill_tokens(
        const DecoderTokenSequence& tokens,
        DecoderLogits* logits) = 0;
    virtual Status prefill_chunk(
        const DecoderTokenSequence& tokens,
        bool start_new_sequence,
        bool produce_logits,
        DecoderLogits* logits) {
        if (!start_new_sequence || !produce_logits) {
            return Status::error("decoder context does not support chunked prefill");
        }
        return prefill_tokens(tokens, logits);
    }
    virtual Status prefill_embeddings(
        const DecoderEmbeddingSequence&,
        DecoderLogits*) {
        return Status::error("decoder context does not support embedding prefill");
    }
    virtual Status decode_token(DecoderTokenId token, DecoderLogits* logits) = 0;
    virtual std::size_t token_count() const = 0;
    virtual DecoderModelContract model_contract() const = 0;
};

}  // namespace cpp_llm_engine::core
