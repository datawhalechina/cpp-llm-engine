// Chapter 01：Decoder 输入装配接口
// 职责：声明 loaded tensors 到 embedding、norm、output 和 decoder layers 的映射。
// 契约：DecoderInput 只描述对象关系，不拥有 tensor payload。
// 验证：chapter01_tensor_loader_test.cpp。
#pragma once

#include "model_config.h"
#include "model_schema.h"
#include "tokenizer_metadata.h"

#include <string>
#include <unordered_map>
#include <vector>

namespace chapter01 {

struct DecoderLayerInput {
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

struct DecoderInput {
    std::string architecture;
    std::string embedding;
    std::string output;
    std::string output_norm;
    std::size_t intermediate_size = 0;
    std::size_t key_value_size = 0;
    std::vector<DecoderLayerInput> layers;

    bool output_tied_to_embedding = false;

    static DecoderInput from_layout(const DenseDecoderTensorLayout& layout);
    bool valid() const;
};

struct LoadedModel {
    ModelConfig config;
    TokenizerMetadata tokenizer;
    DenseDecoderTensorLayout layout;
    DecoderInput decoder;
    std::unordered_map<std::string, Tensor> tensors;
};

class ModelBuilder {
public:
    static bool build(const ModelMetadata& metadata,
                      const TokenizerMetadata& tokenizer,
                      std::unordered_map<std::string, Tensor> tensors,
                      LoadedModel* model,
                      std::string* error = nullptr);
};

}  // namespace chapter01
