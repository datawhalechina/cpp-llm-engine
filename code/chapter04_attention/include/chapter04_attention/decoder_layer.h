// Chapter 04：DecoderLayer 接口
// 职责：按 Pre-LN -> Attention -> Residual -> FFN -> Residual 组合单层结构。
// 契约：输入输出保持 [tokens, model_dim]，layer_index 只标识层位置。
// 验证：chapter04_attention_test.cpp 和 main_attention.cpp。
#pragma once

#include "chapter04_attention/attention.h"
#include "chapter04_attention/feed_forward.h"
#include "chapter04_attention/layer_norm.h"
#include "cpp_llm_engine/core/decoder_contract.h"

#include <cstddef>

namespace chapter04_attention {

class DecoderLayer final {
public:
    DecoderLayer(std::size_t model_dim = 0, std::size_t num_heads = 0, std::size_t ffn_dim = 0, std::size_t layer_index = 0);

    Tensor forward(const Tensor& input) const;
    cpp_llm_engine::core::Status try_forward(
        const cpp_llm_engine::core::TensorViewRef& input,
        const cpp_llm_engine::core::MutableTensorViewRef& output) const;
    bool ready() const;

    std::size_t layer_index() const;
    cpp_llm_engine::core::DecoderLayerContract contract() const;

private:
    std::size_t layer_index_ = 0;
    LayerNorm pre_attn_norm_;
    MultiHeadAttention attention_;
    LayerNorm pre_ffn_norm_;
    FeedForward feed_forward_;
};

}  // namespace chapter04_attention
