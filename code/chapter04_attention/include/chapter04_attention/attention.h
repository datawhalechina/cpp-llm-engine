// Chapter 04：MultiHeadAttention 接口
// 职责：构造 Q/K/V projection，调用 Attention backend 并合并输出 projection。
// 契约：输入为 [tokens, model_dim]，model_dim 必须可整除 num_heads。
// 验证：chapter04_attention_test.cpp。
#pragma once

#include "chapter04_attention/linear_layer.h"
#include "cpp_llm_engine/core/attention_contract.h"

#include <cstddef>

namespace chapter04_attention {

class MultiHeadAttention final {
public:
    MultiHeadAttention(std::size_t model_dim = 0, std::size_t num_heads = 0);
    MultiHeadAttention(std::size_t model_dim,
                       std::size_t num_heads,
                       const cpp_llm_engine::core::AttentionBackend& backend);

    Tensor forward(const Tensor& input) const;
    cpp_llm_engine::core::Status try_forward(
        const cpp_llm_engine::core::TensorViewRef& input,
        const cpp_llm_engine::core::MutableTensorViewRef& output) const;
    bool ready() const;

    std::size_t model_dim() const;
    std::size_t num_heads() const;
    cpp_llm_engine::core::BackendKind backend_kind() const;

private:
    std::size_t model_dim_ = 0;
    std::size_t num_heads_ = 0;
    LinearLayer q_proj_;
    LinearLayer k_proj_;
    LinearLayer v_proj_;
    LinearLayer out_proj_;
    const cpp_llm_engine::core::AttentionBackend* backend_ = nullptr;
};

}  // namespace chapter04_attention
