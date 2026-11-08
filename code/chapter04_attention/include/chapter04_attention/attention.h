// Chapter 04：教学 Multi-Head Attention 接口
// 输入为 [tokens, model_dim]，依次执行 Q/K/V 投影、Attention 和输出投影。
// 对象不拥有注入的 backend；backend 必须比当前对象存活更久。
// 验证：chapter04_attention_test.cpp。
#pragma once

#include "chapter04_attention/kv_cache.h"
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
    MultiHeadAttention(
        std::size_t,
        std::size_t,
        cpp_llm_engine::core::AttentionBackend&&) = delete;

    Tensor forward(const Tensor& input) const;
    cpp_llm_engine::core::Status try_forward(
        const cpp_llm_engine::core::TensorViewRef& input,
        const cpp_llm_engine::core::MutableTensorViewRef& output) const;
    cpp_llm_engine::core::Status try_forward_with_cache(
        const cpp_llm_engine::core::TensorViewRef& input,
        std::size_t query_position_offset,
        KvCache* cache,
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
