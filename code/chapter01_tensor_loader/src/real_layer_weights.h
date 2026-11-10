// Chapter 01：声明真实稠密 decoder layer 权重装配器。
// 职责：按 ModelSchema 从 GGUF 按需加载一层权重，不复制整个模型到内存。
// 契约：只装配 tensor payload 和布局，不执行 Attention/FFN forward。
#pragma once

#include "gguf_reader.h"
#include "model_schema.h"
#include "tensor.h"

#include <cstddef>
#include <string>

namespace chapter01 {

struct RealDenseLayerWeights {
    std::size_t index = 0;
    Tensor attn_norm;
    Tensor attn_q;
    Tensor attn_k;
    Tensor attn_v;
    Tensor attn_output;
    Tensor attn_q_norm;
    Tensor attn_k_norm;
    Tensor ffn_norm;
    Tensor ffn_gate;
    Tensor ffn_up;
    Tensor ffn_down;

    bool valid() const;
};

class RealDenseLayerLoader final {
public:
    static bool load(
        const std::string& model_path,
        const GgufModelIndex& index,
        const DenseDecoderTensorLayout& layout,
        std::size_t layer_index,
        RealDenseLayerWeights* output,
        std::string* error = nullptr);
};

struct RealMoeLayerWeights {
    std::size_t index = 0;
    Tensor attn_norm, attn_q, attn_k, attn_v;
    Tensor attn_q_bias, attn_k_bias, attn_v_bias, attn_output;
    Tensor ffn_norm, router, expert_gate, expert_up, expert_down;
    Tensor shared_expert_router, shared_expert_gate;
    Tensor shared_expert_up, shared_expert_down;
    bool valid() const;
};

class RealMoeLayerLoader final {
public:
    static bool load(const std::string& model_path,
                     const GgufModelIndex& index,
                     const MoeDecoderTensorLayout& layout,
                     std::size_t layer_index,
                     RealMoeLayerWeights* output,
                     std::string* error = nullptr);
};

struct RealDeepSeek2DenseLayerWeights {
    std::size_t index = 0;
    Tensor attn_norm;
    Tensor attn_q;
    Tensor attn_kv_a_mqa;
    Tensor attn_kv_a_norm;
    Tensor attn_kv_b;
    Tensor attn_k_b;
    Tensor attn_v_b;
    Tensor attn_output;
    Tensor ffn_norm;
    Tensor ffn_gate;
    Tensor ffn_up;
    Tensor ffn_down;

    bool uses_combined_kv_b() const;
    bool valid() const;
};

class RealDeepSeek2DenseLayerLoader final {
public:
    static bool load(
        const std::string& model_path,
        const GgufModelIndex& index,
        const DeepSeek2DecoderTensorLayout& layout,
        std::size_t layer_index,
        RealDeepSeek2DenseLayerWeights* output,
        std::string* error = nullptr);
};

struct RealDeepSeek2MoeLayerWeights {
    std::size_t index = 0;
    Tensor attn_norm, attn_q, attn_kv_a_mqa, attn_kv_a_norm;
    Tensor attn_kv_b, attn_k_b, attn_v_b, attn_output;
    Tensor ffn_norm, router, expert_gate, expert_up, expert_down;
    Tensor shared_expert_gate, shared_expert_up, shared_expert_down;

    bool uses_combined_kv_b() const;
    bool valid() const;
};

class RealDeepSeek2MoeLayerLoader final {
public:
    static bool load(
        const std::string& model_path,
        const GgufModelIndex& index,
        const DeepSeek2DecoderTensorLayout& layout,
        std::size_t layer_index,
        RealDeepSeek2MoeLayerWeights* output,
        std::string* error = nullptr);
};

// 迁移期兼容名称；新代码使用 RealDense* 表达多架构层权重。
using RealLlamaLayerWeights = RealDenseLayerWeights;
using RealLlamaLayerLoader = RealDenseLayerLoader;

}  // namespace chapter01
