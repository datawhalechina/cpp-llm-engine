// Chapter 01：Decoder 输入装配实现
// 职责：将已验证的稠密 Decoder tensor layout 组装成稳定映射。
// 契约：缺失权重或层级映射不完整时返回失败，不执行 Transformer 计算。
// 验证：chapter01_tensor_loader_test.cpp 与 gguf_inspect --validate-schema。
#include "model_builder.h"

namespace chapter01 {
namespace {

bool set_error(std::string* error, const std::string& message) {
    if (error != nullptr) *error = message;
    return false;
}

}  // namespace

DecoderInput DecoderInput::from_layout(const DenseDecoderTensorLayout& layout) {
    DecoderInput input;
    input.architecture = layout.architecture;
    input.embedding = layout.embedding;
    input.output = layout.output;
    input.output_norm = layout.output_norm;
    input.output_tied_to_embedding = layout.output_tied_to_embedding;
    input.intermediate_size = layout.intermediate_size;
    input.key_value_size = layout.key_value_size;
    input.layers.reserve(layout.layers.size());
    for (const auto& layer : layout.layers) {
        input.layers.push_back(DecoderLayerInput{
            layer.index,
            layer.attn_norm,
            layer.attn_q,
            layer.attn_k,
            layer.attn_v,
            layer.attn_output,
            layer.attn_q_norm,
            layer.attn_k_norm,
            layer.ffn_norm,
            layer.ffn_gate,
            layer.ffn_up,
            layer.ffn_down,
        });
    }
    return input;
}

bool DecoderInput::valid() const {
    if (architecture.empty() || embedding.empty() || output.empty() || output_norm.empty() ||
        intermediate_size == 0 || key_value_size == 0 || layers.empty()) {
        return false;
    }
    for (const auto& layer : layers) {
        if (layer.attn_norm.empty() || layer.attn_q.empty() || layer.attn_k.empty() ||
            layer.attn_v.empty() || layer.attn_output.empty() || layer.ffn_norm.empty() ||
            layer.ffn_gate.empty() || layer.ffn_up.empty() || layer.ffn_down.empty()) {
            return false;
        }
    }
    return true;
}

bool ModelBuilder::build(const ModelMetadata& metadata,
                         const TokenizerMetadata& tokenizer,
                         std::unordered_map<std::string, Tensor> tensors,
                         LoadedModel* model,
                         std::string* error) {
    if (model == nullptr) return set_error(error, "loaded model output must not be null");
    if (!metadata.valid()) return set_error(error, "model metadata is invalid");
    if (!tokenizer.valid()) return set_error(error, "tokenizer metadata is invalid");
    if (tokenizer.tokens.size() != metadata.vocab_size) {
        return set_error(error, "tokenizer vocabulary size does not match model metadata");
    }

    DenseDecoderTensorLayout layout;
    if (!ModelSchema::validate_tensors(metadata, tensors, &layout, error)) return false;

    LoadedModel built;
    built.config = ModelConfig::from_metadata(metadata);
    built.tokenizer = tokenizer;
    built.layout = std::move(layout);
    built.decoder = DecoderInput::from_layout(built.layout);
    if (!built.decoder.valid()) return set_error(error, "decoder input assembly failed");
    built.tensors = std::move(tensors);
    *model = std::move(built);
    if (error != nullptr) error->clear();
    return true;
}

}  // namespace chapter01
