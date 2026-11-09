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

// 迁移期兼容名称；新代码使用 RealDense* 表达多架构层权重。
using RealLlamaLayerWeights = RealDenseLayerWeights;
using RealLlamaLayerLoader = RealDenseLayerLoader;

}  // namespace chapter01
