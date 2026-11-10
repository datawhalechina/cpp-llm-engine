// Chapter 01：稠密 Decoder tensor schema 接口
// 职责：定义 embedding、norm、output 和 decoder layer tensor 的稳定映射。
// 契约：validate_gguf 成功后才能构造 DecoderInput。
// 验证：chapter01_tensor_loader_test.cpp。
#pragma once

#include "gguf_reader.h"
#include "model_metadata.h"
#include "tensor.h"

#include <cstddef>
#include <string>
#include <unordered_map>
#include <vector>

namespace chapter01 {

struct DecoderLayerLayout {
    std::size_t index = 0;
    std::string attn_norm;
    std::string attn_q;
    std::string attn_k;
    std::string attn_v;
    std::string attn_output;
    std::string attn_q_norm;
    std::string attn_k_norm;
    std::string ffn_norm;
    std::string ffn_gate;
    std::string ffn_up;
    std::string ffn_down;
};

struct DenseDecoderTensorLayout {
    std::string architecture;
    std::string embedding;
    std::string output;
    std::string output_norm;
    bool output_tied_to_embedding = false;
    std::size_t intermediate_size = 0;
    std::size_t key_value_size = 0;
    std::vector<DecoderLayerLayout> layers;
};

struct MoeDecoderLayerLayout {
    std::size_t index = 0;
    std::string attn_norm;
    std::string attn_q;
    std::string attn_k;
    std::string attn_v;
    std::string attn_q_bias;
    std::string attn_k_bias;
    std::string attn_v_bias;
    std::string attn_output;
    std::string ffn_norm;
    std::string router;
    std::string expert_gate;
    std::string expert_up;
    std::string expert_down;
    std::string shared_expert_router;
    std::string shared_expert_gate;
    std::string shared_expert_up;
    std::string shared_expert_down;
};

struct MoeDecoderTensorLayout {
    std::string architecture;
    std::string embedding;
    std::string output;
    std::string output_norm;
    std::size_t expert_count = 0;
    std::size_t expert_used_count = 0;
    std::vector<MoeDecoderLayerLayout> layers;
};

struct DeepSeek2DecoderLayerLayout {
    std::size_t index = 0;
    bool uses_dense_ffn = false;
    std::string attn_norm;
    std::string attn_q;
    std::string attn_kv_a_mqa;
    std::string attn_kv_a_norm;
    std::string attn_kv_b;
    std::string attn_k_b;
    std::string attn_v_b;
    std::string attn_output;
    std::string ffn_norm;
    std::string ffn_gate;
    std::string ffn_up;
    std::string ffn_down;
    std::string router;
    std::string expert_gate;
    std::string expert_up;
    std::string expert_down;
    std::string shared_expert_gate;
    std::string shared_expert_up;
    std::string shared_expert_down;
};

struct DeepSeek2DecoderTensorLayout {
    std::string architecture;
    std::string embedding;
    std::string output;
    std::string output_norm;
    bool output_tied_to_embedding = false;
    std::size_t leading_dense_layer_count = 0;
    std::size_t expert_count = 0;
    std::size_t expert_used_count = 0;
    std::vector<DeepSeek2DecoderLayerLayout> layers;
};

// 迁移期兼容名称；新代码使用 DenseDecoderTensorLayout 表达多架构布局。
using LlamaTensorLayout = DenseDecoderTensorLayout;

class ModelSchema final {
public:
    static bool validate_gguf(const ModelMetadata& metadata,
                              const GgufModelIndex& index,
                              DenseDecoderTensorLayout* layout,
                              std::string* error = nullptr);

    static bool validate_tensors(const ModelMetadata& metadata,
                                 const std::unordered_map<std::string, Tensor>& tensors,
                                 DenseDecoderTensorLayout* layout,
                                 std::string* error = nullptr);

    static bool validate_moe_gguf(const ModelMetadata& metadata,
                                  const GgufModelIndex& index,
                                  MoeDecoderTensorLayout* layout,
                                  std::string* error = nullptr);

    static bool validate_deepseek2_gguf(const ModelMetadata& metadata,
                                        const GgufModelIndex& index,
                                        DeepSeek2DecoderTensorLayout* layout,
                                        std::string* error = nullptr);
};

}  // namespace chapter01
