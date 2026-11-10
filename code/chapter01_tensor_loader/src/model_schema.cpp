// Chapter 01：稠密 Decoder tensor schema 校验实现
// 职责：检查 GGUF tensor names、shape、层数和架构附加张量。
// 契约：schema 只消费 index 和 metadata，不加载完整权重。
// 验证：chapter01_tensor_loader_test.cpp 与 gguf_inspect --validate-schema。
#include "model_schema.h"

#include <limits>
#include <sstream>
#include <unordered_map>

namespace chapter01 {
namespace {

struct ShapeDescriptor {
    std::string name;
    std::vector<std::size_t> shape;
    bool has_payload = true;
};

bool fail(std::string* error, const std::string& message) {
    if (error != nullptr) *error = message;
    return false;
}

std::string layer_name(std::size_t index, const char* suffix) {
    return "blk." + std::to_string(index) + "." + suffix;
}

DecoderLayerLayout make_layer(std::size_t index, const std::string& architecture) {
    DecoderLayerLayout layer;
    layer.index = index;
    layer.attn_norm = layer_name(index, "attn_norm.weight");
    layer.attn_q = layer_name(index, "attn_q.weight");
    layer.attn_k = layer_name(index, "attn_k.weight");
    layer.attn_v = layer_name(index, "attn_v.weight");
    layer.attn_output = layer_name(index, "attn_output.weight");
    if (architecture == "qwen3") {
        layer.attn_q_norm = layer_name(index, "attn_q_norm.weight");
        layer.attn_k_norm = layer_name(index, "attn_k_norm.weight");
    }
    layer.ffn_norm = layer_name(index, "ffn_norm.weight");
    layer.ffn_gate = layer_name(index, "ffn_gate.weight");
    layer.ffn_up = layer_name(index, "ffn_up.weight");
    layer.ffn_down = layer_name(index, "ffn_down.weight");
    return layer;
}

MoeDecoderLayerLayout make_moe_layer(std::size_t index) {
    MoeDecoderLayerLayout layer;
    layer.index = index;
    layer.attn_norm = layer_name(index, "attn_norm.weight");
    layer.attn_q = layer_name(index, "attn_q.weight");
    layer.attn_k = layer_name(index, "attn_k.weight");
    layer.attn_v = layer_name(index, "attn_v.weight");
    layer.attn_q_bias = layer_name(index, "attn_q.bias");
    layer.attn_k_bias = layer_name(index, "attn_k.bias");
    layer.attn_v_bias = layer_name(index, "attn_v.bias");
    layer.attn_output = layer_name(index, "attn_output.weight");
    layer.ffn_norm = layer_name(index, "ffn_norm.weight");
    layer.router = layer_name(index, "ffn_gate_inp.weight");
    layer.expert_gate = layer_name(index, "ffn_gate_exps.weight");
    layer.expert_up = layer_name(index, "ffn_up_exps.weight");
    layer.expert_down = layer_name(index, "ffn_down_exps.weight");
    layer.shared_expert_router = layer_name(index, "ffn_gate_inp_shexp.weight");
    layer.shared_expert_gate = layer_name(index, "ffn_gate_shexp.weight");
    layer.shared_expert_up = layer_name(index, "ffn_up_shexp.weight");
    layer.shared_expert_down = layer_name(index, "ffn_down_shexp.weight");
    return layer;
}

bool checked_dimension(std::uint64_t value, std::size_t* result, std::string* error) {
    if (value > std::numeric_limits<std::size_t>::max()) {
        return fail(error, "GGUF tensor dimension exceeds host size");
    }
    *result = static_cast<std::size_t>(value);
    return true;
}

bool checked_product(std::size_t left,
                     std::size_t right,
                     std::size_t* result,
                     std::string* error) {
    if (left != 0 && right > std::numeric_limits<std::size_t>::max() / left) {
        return fail(error, "attention projection dimension exceeds host size");
    }
    *result = left * right;
    return true;
}

bool validate_shapes(const ModelMetadata& metadata,
                     const std::vector<ShapeDescriptor>& descriptors,
                     DenseDecoderTensorLayout* layout,
                     std::string* error) {
    if (layout == nullptr) return fail(error, "model layout output must not be null");
    if (metadata.architecture != "llama" && metadata.architecture != "qwen3") {
        return fail(error, "dense tensor schema supports llama and qwen3 architectures");
    }

    std::unordered_map<std::string, const ShapeDescriptor*> by_name;
    for (const auto& descriptor : descriptors) {
        if (!by_name.emplace(descriptor.name, &descriptor).second) {
            return fail(error, "duplicate tensor in model schema: " + descriptor.name);
        }
    }

    const auto find = [&](const std::string& name) -> const ShapeDescriptor* {
        const auto it = by_name.find(name);
        return it == by_name.end() ? nullptr : it->second;
    };
    const auto expect = [&](const std::string& name,
                            const std::vector<std::size_t>& shape) -> bool {
        const auto* descriptor = find(name);
        const std::string family = metadata.architecture == "llama" ? "Llama" : "Qwen3";
        if (descriptor == nullptr) return fail(error, "missing " + family + " tensor: " + name);
        if (!descriptor->has_payload) return fail(error, family + " tensor has no payload: " + name);
        if (descriptor->shape != shape) {
            std::ostringstream message;
            message << "shape mismatch for " << name;
            return fail(error, message.str());
        }
        return true;
    };

    if (!expect("token_embd.weight", {metadata.hidden_size, metadata.vocab_size}) ||
        !expect("output_norm.weight", {metadata.hidden_size})) {
        return false;
    }

    DenseDecoderTensorLayout built;
    built.architecture = metadata.architecture;
    built.embedding = "token_embd.weight";
    built.output_norm = "output_norm.weight";
    if (find("output.weight") != nullptr) {
        if (!expect("output.weight", {metadata.hidden_size, metadata.vocab_size})) return false;
        built.output = "output.weight";
    } else if (metadata.architecture == "qwen3") {
        built.output = built.embedding;
        built.output_tied_to_embedding = true;
    } else {
        return fail(error, "missing Llama tensor: output.weight");
    }
    built.layers.reserve(metadata.layer_count);

    std::size_t query_size = metadata.hidden_size;
    std::size_t key_size = 0;
    std::size_t value_size = 0;
    std::size_t output_input_size = metadata.hidden_size;
    std::size_t qk_norm_size = 0;
    if (metadata.architecture == "qwen3") {
        if (metadata.attention_head_count == 0 ||
            metadata.attention_head_count_kv == 0 ||
            metadata.attention_key_length == 0 ||
            metadata.attention_value_length == 0) {
            return fail(error, "Qwen3 attention head count and head length metadata are required");
        }
        if (metadata.attention_key_length != metadata.attention_value_length) {
            return fail(error, "Qwen3 key and value head lengths must match");
        }
        if (!checked_product(metadata.attention_key_length,
                             metadata.attention_head_count, &query_size, error) ||
            !checked_product(metadata.attention_key_length,
                             metadata.attention_head_count_kv, &key_size, error) ||
            !checked_product(metadata.attention_value_length,
                             metadata.attention_head_count_kv, &value_size, error) ||
            !checked_product(metadata.attention_value_length,
                             metadata.attention_head_count, &output_input_size, error)) {
            return false;
        }
        qk_norm_size = metadata.attention_key_length;
    }

    std::size_t intermediate_size = 0;
    std::size_t key_value_size = 0;
    for (std::size_t i = 0; i < metadata.layer_count; ++i) {
        const auto layer = make_layer(i, metadata.architecture);
        if (!expect(layer.attn_norm, {metadata.hidden_size}) ||
            !expect(layer.ffn_norm, {metadata.hidden_size}) ||
            !expect(layer.attn_q, {metadata.hidden_size, query_size}) ||
            !expect(layer.attn_output, {output_input_size, metadata.hidden_size})) {
            return false;
        }
        if (metadata.architecture == "qwen3" &&
            (!expect(layer.attn_q_norm, {qk_norm_size}) ||
             !expect(layer.attn_k_norm, {qk_norm_size}))) {
            return false;
        }

        const auto* gate = find(layer.ffn_gate);
        const auto* up = find(layer.ffn_up);
        const auto* down = find(layer.ffn_down);
        const auto* key = find(layer.attn_k);
        const auto* value = find(layer.attn_v);
        if (gate == nullptr || up == nullptr || down == nullptr || key == nullptr || value == nullptr) {
            return fail(error, "Llama layer is missing FFN or KV projection tensor");
        }
        if (gate->shape.size() != 2 || up->shape != gate->shape ||
            down->shape.size() != 2 || down->shape[0] != gate->shape[1] ||
            down->shape[1] != metadata.hidden_size) {
            return fail(error, "FFN tensor shapes do not match Llama layer contract");
        }
        if (key->shape.size() != 2 || value->shape.size() != 2 ||
            key->shape[0] != metadata.hidden_size ||
            value->shape[0] != metadata.hidden_size) {
            return fail(error, "attention K/V tensor shapes do not match decoder layer contract");
        }
        if (metadata.architecture == "qwen3" &&
            (key->shape[1] != key_size || value->shape[1] != value_size)) {
            return fail(error, "attention K/V tensor shapes do not match Qwen3 head metadata");
        }
        if (intermediate_size == 0) intermediate_size = gate->shape[1];
        if (key_value_size == 0) key_value_size = key->shape[1];
        if (intermediate_size != gate->shape[1] || key_value_size != key->shape[1] ||
            value->shape[1] != key_value_size) {
            return fail(error, "decoder layer dimensions are inconsistent");
        }
        for (const auto* descriptor : {gate, up, down, key, value}) {
            if (!descriptor->has_payload) {
                return fail(error, "decoder tensor has no payload: " + descriptor->name);
            }
        }
        built.layers.push_back(layer);
    }
    built.intermediate_size = intermediate_size;
    built.key_value_size = key_value_size;
    *layout = std::move(built);
    if (error != nullptr) error->clear();
    return true;
}

bool validate_moe_shapes(const ModelMetadata& metadata,
                         const std::vector<ShapeDescriptor>& descriptors,
                         MoeDecoderTensorLayout* layout,
                         std::string* error) {
    if (layout == nullptr) return fail(error, "MoE model layout output must not be null");
    if (metadata.architecture != "qwen2moe") {
        return fail(error, "MoE tensor schema supports qwen2moe architecture");
    }
    if (metadata.expert_count == 0 || metadata.expert_used_count == 0 ||
        metadata.expert_used_count > metadata.expert_count ||
        metadata.expert_feed_forward_length == 0 ||
        metadata.expert_shared_feed_forward_length == 0) {
        return fail(error, "Qwen2MoE expert metadata is incomplete or inconsistent");
    }

    std::unordered_map<std::string, const ShapeDescriptor*> by_name;
    for (const auto& descriptor : descriptors) {
        if (!by_name.emplace(descriptor.name, &descriptor).second) {
            return fail(error, "duplicate tensor in MoE model schema: " + descriptor.name);
        }
    }
    const auto expect = [&](const std::string& name,
                            const std::vector<std::size_t>& shape) -> bool {
        const auto it = by_name.find(name);
        if (it == by_name.end()) return fail(error, "missing Qwen2MoE tensor: " + name);
        if (!it->second->has_payload) return fail(error, "Qwen2MoE tensor has no payload: " + name);
        if (it->second->shape != shape) return fail(error, "shape mismatch for " + name);
        return true;
    };

    if (!expect("token_embd.weight", {metadata.hidden_size, metadata.vocab_size}) ||
        !expect("output_norm.weight", {metadata.hidden_size}) ||
        !expect("output.weight", {metadata.hidden_size, metadata.vocab_size})) {
        return false;
    }

    MoeDecoderTensorLayout built;
    built.architecture = metadata.architecture;
    built.embedding = "token_embd.weight";
    built.output = "output.weight";
    built.output_norm = "output_norm.weight";
    built.expert_count = metadata.expert_count;
    built.expert_used_count = metadata.expert_used_count;
    built.layers.reserve(metadata.layer_count);
    for (std::size_t i = 0; i < metadata.layer_count; ++i) {
        const auto layer = make_moe_layer(i);
        if (!expect(layer.attn_norm, {metadata.hidden_size}) ||
            !expect(layer.attn_q, {metadata.hidden_size, metadata.hidden_size}) ||
            !expect(layer.attn_k, {metadata.hidden_size, metadata.hidden_size}) ||
            !expect(layer.attn_v, {metadata.hidden_size, metadata.hidden_size}) ||
            !expect(layer.attn_q_bias, {metadata.hidden_size}) ||
            !expect(layer.attn_k_bias, {metadata.hidden_size}) ||
            !expect(layer.attn_v_bias, {metadata.hidden_size}) ||
            !expect(layer.attn_output, {metadata.hidden_size, metadata.hidden_size}) ||
            !expect(layer.ffn_norm, {metadata.hidden_size}) ||
            !expect(layer.router, {metadata.hidden_size, metadata.expert_count}) ||
            !expect(layer.expert_gate, {metadata.hidden_size,
                                        metadata.expert_feed_forward_length,
                                        metadata.expert_count}) ||
            !expect(layer.expert_up, {metadata.hidden_size,
                                      metadata.expert_feed_forward_length,
                                      metadata.expert_count}) ||
            !expect(layer.expert_down, {metadata.expert_feed_forward_length,
                                        metadata.hidden_size,
                                        metadata.expert_count}) ||
            !expect(layer.shared_expert_router, {metadata.hidden_size}) ||
            !expect(layer.shared_expert_gate, {metadata.hidden_size,
                                               metadata.expert_shared_feed_forward_length}) ||
            !expect(layer.shared_expert_up, {metadata.hidden_size,
                                             metadata.expert_shared_feed_forward_length}) ||
            !expect(layer.shared_expert_down, {metadata.expert_shared_feed_forward_length,
                                               metadata.hidden_size})) {
            return false;
        }
        built.layers.push_back(layer);
    }
    *layout = std::move(built);
    if (error != nullptr) error->clear();
    return true;
}

DeepSeek2DecoderLayerLayout make_deepseek2_layer(std::size_t index,
                                                 bool uses_dense_ffn) {
    DeepSeek2DecoderLayerLayout layer;
    layer.index = index;
    layer.uses_dense_ffn = uses_dense_ffn;
    layer.attn_norm = layer_name(index, "attn_norm.weight");
    layer.attn_q = layer_name(index, "attn_q.weight");
    layer.attn_kv_a_mqa = layer_name(index, "attn_kv_a_mqa.weight");
    layer.attn_kv_a_norm = layer_name(index, "attn_kv_a_norm.weight");
    layer.attn_kv_b = layer_name(index, "attn_kv_b.weight");
    layer.attn_k_b = layer_name(index, "attn_k_b.weight");
    layer.attn_v_b = layer_name(index, "attn_v_b.weight");
    layer.attn_output = layer_name(index, "attn_output.weight");
    layer.ffn_norm = layer_name(index, "ffn_norm.weight");
    if (uses_dense_ffn) {
        layer.ffn_gate = layer_name(index, "ffn_gate.weight");
        layer.ffn_up = layer_name(index, "ffn_up.weight");
        layer.ffn_down = layer_name(index, "ffn_down.weight");
    } else {
        layer.router = layer_name(index, "ffn_gate_inp.weight");
        layer.expert_gate = layer_name(index, "ffn_gate_exps.weight");
        layer.expert_up = layer_name(index, "ffn_up_exps.weight");
        layer.expert_down = layer_name(index, "ffn_down_exps.weight");
        layer.shared_expert_gate = layer_name(index, "ffn_gate_shexp.weight");
        layer.shared_expert_up = layer_name(index, "ffn_up_shexp.weight");
        layer.shared_expert_down = layer_name(index, "ffn_down_shexp.weight");
    }
    return layer;
}

bool validate_deepseek2_shapes(const ModelMetadata& metadata,
                               const std::vector<ShapeDescriptor>& descriptors,
                               DeepSeek2DecoderTensorLayout* layout,
                               std::string* error) {
    if (layout == nullptr) return fail(error, "DeepSeek2 model layout output must not be null");
    if (metadata.architecture != "deepseek2") {
        return fail(error, "DeepSeek2 tensor schema requires deepseek2 architecture");
    }
    if (metadata.attention_head_count == 0 ||
        metadata.attention_kv_lora_rank == 0 ||
        metadata.attention_key_length_mla <= metadata.rope_dimension_count ||
        metadata.attention_value_length_mla == 0 ||
        metadata.attention_key_length !=
            metadata.attention_kv_lora_rank + metadata.rope_dimension_count ||
        metadata.attention_value_length != metadata.attention_kv_lora_rank) {
        return fail(error, "DeepSeek2 MLA metadata is incomplete or inconsistent");
    }
    if (metadata.leading_dense_layer_count > metadata.layer_count ||
        metadata.feed_forward_length == 0 || metadata.expert_count == 0 ||
        metadata.expert_used_count == 0 ||
        metadata.expert_used_count > metadata.expert_count ||
        metadata.expert_feed_forward_length == 0 ||
        metadata.expert_shared_count == 0) {
        return fail(error, "DeepSeek2 dense and expert metadata is incomplete or inconsistent");
    }

    std::unordered_map<std::string, const ShapeDescriptor*> by_name;
    for (const auto& descriptor : descriptors) {
        if (!by_name.emplace(descriptor.name, &descriptor).second) {
            return fail(error, "duplicate tensor in DeepSeek2 schema: " + descriptor.name);
        }
    }
    const auto find = [&](const std::string& name) -> const ShapeDescriptor* {
        const auto it = by_name.find(name);
        return it == by_name.end() ? nullptr : it->second;
    };
    const auto expect = [&](const std::string& name,
                            const std::vector<std::size_t>& shape) -> bool {
        const auto* descriptor = find(name);
        if (descriptor == nullptr) return fail(error, "missing DeepSeek2 tensor: " + name);
        if (!descriptor->has_payload) return fail(error, "DeepSeek2 tensor has no payload: " + name);
        if (descriptor->shape != shape) return fail(error, "shape mismatch for " + name);
        return true;
    };

    if (!expect("token_embd.weight", {metadata.hidden_size, metadata.vocab_size}) ||
        !expect("output_norm.weight", {metadata.hidden_size})) {
        return false;
    }

    DeepSeek2DecoderTensorLayout built;
    built.architecture = metadata.architecture;
    built.embedding = "token_embd.weight";
    built.output_norm = "output_norm.weight";
    if (find("output.weight") != nullptr) {
        if (!expect("output.weight", {metadata.hidden_size, metadata.vocab_size})) return false;
        built.output = "output.weight";
    } else {
        built.output = built.embedding;
        built.output_tied_to_embedding = true;
    }
    built.leading_dense_layer_count = metadata.leading_dense_layer_count;
    built.expert_count = metadata.expert_count;
    built.expert_used_count = metadata.expert_used_count;
    built.layers.reserve(metadata.layer_count);

    const std::size_t qk_nope =
        metadata.attention_key_length_mla - metadata.rope_dimension_count;
    std::size_t query_size = 0;
    std::size_t output_input_size = 0;
    if (!checked_product(metadata.attention_head_count,
                         metadata.attention_key_length_mla, &query_size, error) ||
        !checked_product(metadata.attention_head_count,
                         metadata.attention_value_length_mla,
                         &output_input_size, error)) {
        return false;
    }
    std::size_t shared_ffn_size = 0;
    if (!checked_product(metadata.expert_feed_forward_length,
                         metadata.expert_shared_count, &shared_ffn_size, error)) {
        return false;
    }

    for (std::size_t i = 0; i < metadata.layer_count; ++i) {
        const bool dense = i < metadata.leading_dense_layer_count;
        auto layer = make_deepseek2_layer(i, dense);
        if (!expect(layer.attn_norm, {metadata.hidden_size}) ||
            !expect(layer.attn_q, {metadata.hidden_size, query_size}) ||
            !expect(layer.attn_kv_a_mqa,
                    {metadata.hidden_size,
                     metadata.attention_kv_lora_rank + metadata.rope_dimension_count}) ||
            !expect(layer.attn_kv_a_norm, {metadata.attention_kv_lora_rank}) ||
            !expect(layer.attn_output, {output_input_size, metadata.hidden_size}) ||
            !expect(layer.ffn_norm, {metadata.hidden_size})) {
            return false;
        }
        if (find(layer.attn_k_b) != nullptr || find(layer.attn_v_b) != nullptr) {
            if (!expect(layer.attn_k_b,
                        {qk_nope, metadata.attention_kv_lora_rank,
                         metadata.attention_head_count}) ||
                !expect(layer.attn_v_b,
                        {metadata.attention_kv_lora_rank,
                         metadata.attention_value_length_mla,
                         metadata.attention_head_count})) {
                return false;
            }
            layer.attn_kv_b.clear();
        } else {
            std::size_t expanded_kv_size = 0;
            if (!checked_product(metadata.attention_head_count,
                                 qk_nope + metadata.attention_value_length_mla,
                                 &expanded_kv_size, error) ||
                !expect(layer.attn_kv_b,
                        {metadata.attention_kv_lora_rank, expanded_kv_size})) {
                return false;
            }
            layer.attn_k_b.clear();
            layer.attn_v_b.clear();
        }
        if (dense) {
            if (!expect(layer.ffn_gate,
                        {metadata.hidden_size, metadata.feed_forward_length}) ||
                !expect(layer.ffn_up,
                        {metadata.hidden_size, metadata.feed_forward_length}) ||
                !expect(layer.ffn_down,
                        {metadata.feed_forward_length, metadata.hidden_size})) {
                return false;
            }
        } else if (!expect(layer.router,
                           {metadata.hidden_size, metadata.expert_count}) ||
                   !expect(layer.expert_gate,
                           {metadata.hidden_size,
                            metadata.expert_feed_forward_length,
                            metadata.expert_count}) ||
                   !expect(layer.expert_up,
                           {metadata.hidden_size,
                            metadata.expert_feed_forward_length,
                            metadata.expert_count}) ||
                   !expect(layer.expert_down,
                           {metadata.expert_feed_forward_length,
                            metadata.hidden_size,
                            metadata.expert_count}) ||
                   !expect(layer.shared_expert_gate,
                           {metadata.hidden_size, shared_ffn_size}) ||
                   !expect(layer.shared_expert_up,
                           {metadata.hidden_size, shared_ffn_size}) ||
                   !expect(layer.shared_expert_down,
                           {shared_ffn_size, metadata.hidden_size})) {
            return false;
        }
        built.layers.push_back(layer);
    }
    *layout = std::move(built);
    if (error != nullptr) error->clear();
    return true;
}

}  // namespace

bool ModelSchema::validate_gguf(const ModelMetadata& metadata,
                                const GgufModelIndex& index,
                                DenseDecoderTensorLayout* layout,
                                std::string* error) {
    std::vector<ShapeDescriptor> descriptors;
    descriptors.reserve(index.tensors.size());
    for (const auto& tensor : index.tensors) {
        ShapeDescriptor descriptor;
        descriptor.name = tensor.name;
        descriptor.shape.reserve(tensor.dimensions.size());
        for (const auto dimension : tensor.dimensions) {
            std::size_t converted = 0;
            if (!checked_dimension(dimension, &converted, error)) return false;
            descriptor.shape.push_back(converted);
        }
        descriptors.push_back(std::move(descriptor));
    }
    return validate_shapes(metadata, descriptors, layout, error);
}

bool ModelSchema::validate_tensors(const ModelMetadata& metadata,
                                   const std::unordered_map<std::string, Tensor>& tensors,
                                   DenseDecoderTensorLayout* layout,
                                   std::string* error) {
    std::vector<ShapeDescriptor> descriptors;
    descriptors.reserve(tensors.size());
    for (const auto& [name, tensor] : tensors) {
        ShapeDescriptor descriptor;
        descriptor.name = name;
        descriptor.has_payload = tensor.bytes() > 0 && tensor.raw_data() != nullptr;
        descriptor.shape.reserve(tensor.shape().size());
        for (const auto dimension : tensor.shape()) {
            if (dimension < 0) return fail(error, "tensor shape contains a negative dimension: " + name);
            descriptor.shape.push_back(static_cast<std::size_t>(dimension));
        }
        descriptors.push_back(std::move(descriptor));
    }
    return validate_shapes(metadata, descriptors, layout, error);
}

bool ModelSchema::validate_moe_gguf(const ModelMetadata& metadata,
                                    const GgufModelIndex& index,
                                    MoeDecoderTensorLayout* layout,
                                    std::string* error) {
    std::vector<ShapeDescriptor> descriptors;
    descriptors.reserve(index.tensors.size());
    for (const auto& tensor : index.tensors) {
        ShapeDescriptor descriptor;
        descriptor.name = tensor.name;
        descriptor.shape.reserve(tensor.dimensions.size());
        for (const auto dimension : tensor.dimensions) {
            std::size_t converted = 0;
            if (!checked_dimension(dimension, &converted, error)) return false;
            descriptor.shape.push_back(converted);
        }
        descriptors.push_back(std::move(descriptor));
    }
    return validate_moe_shapes(metadata, descriptors, layout, error);
}

bool ModelSchema::validate_deepseek2_gguf(
    const ModelMetadata& metadata,
    const GgufModelIndex& index,
    DeepSeek2DecoderTensorLayout* layout,
    std::string* error) {
    std::vector<ShapeDescriptor> descriptors;
    descriptors.reserve(index.tensors.size());
    for (const auto& tensor : index.tensors) {
        ShapeDescriptor descriptor;
        descriptor.name = tensor.name;
        descriptor.shape.reserve(tensor.dimensions.size());
        for (const auto dimension : tensor.dimensions) {
            std::size_t converted = 0;
            if (!checked_dimension(dimension, &converted, error)) return false;
            descriptor.shape.push_back(converted);
        }
        descriptors.push_back(std::move(descriptor));
    }
    return validate_deepseek2_shapes(metadata, descriptors, layout, error);
}

}  // namespace chapter01
