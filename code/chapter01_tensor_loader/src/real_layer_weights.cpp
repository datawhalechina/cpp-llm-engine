// Chapter 01：实现真实稠密 decoder layer 权重按需装配。
// 契约：层布局来自 ModelSchema，loader 只读取指定层，不执行模型计算。
// 验证：gguf_first_layer_inspect 使用真实 TinyLlama 检查九个 layer-0 tensor。
#include "real_layer_weights.h"

#include "gguf_tensor_loader.h"

#include <sstream>
#include <utility>

namespace chapter01 {
namespace {

void set_error(std::string* error, const std::string& message) {
    if (error != nullptr) *error = message;
}

const GgufTensorInfo* find_tensor(
    const GgufModelIndex& index,
    const std::string& name) {
    for (const auto& tensor : index.tensors) {
        if (tensor.name == name) return &tensor;
    }
    return nullptr;
}

bool load_named(
    const std::string& model_path,
    const GgufModelIndex& index,
    const std::string& name,
    Tensor* output,
    std::string* error) {
    const auto* info = find_tensor(index, name);
    if (info == nullptr) {
        set_error(error, "missing layer tensor: " + name);
        return false;
    }
    std::string load_error;
    if (!GgufTensorLoader::load_tensor(
            model_path, index, *info, output, &load_error)) {
        set_error(error, "failed to load " + name + ": " + load_error);
        return false;
    }
    if (error != nullptr) error->clear();
    return true;
}

bool load_optional_named(
    const std::string& model_path,
    const GgufModelIndex& index,
    const std::string& name,
    Tensor* output,
    std::string* error) {
    return name.empty() || load_named(model_path, index, name, output, error);
}

}  // namespace

bool RealDenseLayerWeights::valid() const {
    const bool qk_norms_match =
        (attn_q_norm.bytes() == 0) == (attn_k_norm.bytes() == 0);
    return qk_norms_match && attn_norm.bytes() > 0 && attn_q.bytes() > 0 && attn_k.bytes() > 0 &&
            attn_v.bytes() > 0 && attn_output.bytes() > 0 && ffn_norm.bytes() > 0 &&
            ffn_gate.bytes() > 0 && ffn_up.bytes() > 0 && ffn_down.bytes() > 0;
}

bool RealDenseLayerLoader::load(
    const std::string& model_path,
    const GgufModelIndex& index,
    const DenseDecoderTensorLayout& layout,
    std::size_t layer_index,
    RealDenseLayerWeights* output,
    std::string* error) {
    if (output == nullptr) {
        set_error(error, "layer weights output must not be null");
        return false;
    }
    if (layer_index >= layout.layers.size()) {
        set_error(error, "layer index is outside model layout");
        return false;
    }

    const auto& names = layout.layers[layer_index];
    RealDenseLayerWeights loaded;
    loaded.index = layer_index;
    if (!load_named(model_path, index, names.attn_norm, &loaded.attn_norm, error) ||
        !load_named(model_path, index, names.attn_q, &loaded.attn_q, error) ||
        !load_named(model_path, index, names.attn_k, &loaded.attn_k, error) ||
        !load_named(model_path, index, names.attn_v, &loaded.attn_v, error) ||
        !load_named(model_path, index, names.attn_output, &loaded.attn_output, error) ||
        !load_optional_named(model_path, index, names.attn_q_norm,
                             &loaded.attn_q_norm, error) ||
        !load_optional_named(model_path, index, names.attn_k_norm,
                             &loaded.attn_k_norm, error) ||
        !load_named(model_path, index, names.ffn_norm, &loaded.ffn_norm, error) ||
        !load_named(model_path, index, names.ffn_gate, &loaded.ffn_gate, error) ||
        !load_named(model_path, index, names.ffn_up, &loaded.ffn_up, error) ||
        !load_named(model_path, index, names.ffn_down, &loaded.ffn_down, error)) {
        return false;
    }
    if (!loaded.valid()) {
        set_error(error, "loaded dense decoder layer has an empty tensor payload");
        return false;
    }
    *output = std::move(loaded);
    if (error != nullptr) error->clear();
    return true;
}

bool RealMoeLayerWeights::valid() const {
    return attn_norm.bytes() && attn_q.bytes() && attn_k.bytes() && attn_v.bytes() &&
        attn_q_bias.bytes() && attn_k_bias.bytes() && attn_v_bias.bytes() &&
        attn_output.bytes() && ffn_norm.bytes() && router.bytes() &&
        expert_gate.bytes() && expert_up.bytes() && expert_down.bytes() &&
        shared_expert_router.bytes() && shared_expert_gate.bytes() &&
        shared_expert_up.bytes() && shared_expert_down.bytes();
}

bool RealMoeLayerLoader::load(const std::string& model_path,
                              const GgufModelIndex& index,
                              const MoeDecoderTensorLayout& layout,
                              std::size_t layer_index,
                              RealMoeLayerWeights* output,
                              std::string* error) {
    if (output == nullptr || layer_index >= layout.layers.size()) {
        set_error(error, output == nullptr ? "MoE layer output must not be null" :
                                            "MoE layer index is outside model layout");
        return false;
    }
    const auto& n = layout.layers[layer_index];
    RealMoeLayerWeights w;
    w.index = layer_index;
    if (!load_named(model_path,index,n.attn_norm,&w.attn_norm,error) ||
        !load_named(model_path,index,n.attn_q,&w.attn_q,error) ||
        !load_named(model_path,index,n.attn_k,&w.attn_k,error) ||
        !load_named(model_path,index,n.attn_v,&w.attn_v,error) ||
        !load_named(model_path,index,n.attn_q_bias,&w.attn_q_bias,error) ||
        !load_named(model_path,index,n.attn_k_bias,&w.attn_k_bias,error) ||
        !load_named(model_path,index,n.attn_v_bias,&w.attn_v_bias,error) ||
        !load_named(model_path,index,n.attn_output,&w.attn_output,error) ||
        !load_named(model_path,index,n.ffn_norm,&w.ffn_norm,error) ||
        !load_named(model_path,index,n.router,&w.router,error) ||
        !load_named(model_path,index,n.expert_gate,&w.expert_gate,error) ||
        !load_named(model_path,index,n.expert_up,&w.expert_up,error) ||
        !load_named(model_path,index,n.expert_down,&w.expert_down,error) ||
        !load_named(model_path,index,n.shared_expert_router,&w.shared_expert_router,error) ||
        !load_named(model_path,index,n.shared_expert_gate,&w.shared_expert_gate,error) ||
        !load_named(model_path,index,n.shared_expert_up,&w.shared_expert_up,error) ||
        !load_named(model_path,index,n.shared_expert_down,&w.shared_expert_down,error)) return false;
    if (!w.valid()) { set_error(error, "loaded MoE decoder layer has an empty tensor payload"); return false; }
    *output = std::move(w);
    if (error != nullptr) error->clear();
    return true;
}

}  // namespace chapter01
