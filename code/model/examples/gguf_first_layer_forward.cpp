// Chapter 04/05：真实 TinyLlama 首层单 token CPU forward。
// 职责：把 Chapter01 的 GGUF 权重、Chapter04 的层结构和 Chapter05 的量化 Linear 连接起来。
// 限制：只执行 layer-0 的单 token，不代表完整模型生成或 llama.cpp 数值等价。
#include "gguf_reader.h"
#include "gguf_tensor_loader.h"
#include "model_metadata_adapter.h"
#include "model_schema.h"
#include "real_layer_weights.h"

#include "chapter05_quantization/q4k_dequantize.h"
#include "chapter05_quantization/q4k_linear.h"
#include "chapter05_quantization/q6k_linear.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace {

float half_to_float(std::uint16_t bits) {
    const std::uint32_t sign = (bits & 0x8000u) << 16;
    const std::uint32_t exponent = (bits >> 10) & 0x1fu;
    const std::uint32_t mantissa = bits & 0x03ffu;
    std::uint32_t result = 0;
    if (exponent == 0) {
        if (mantissa == 0) {
            result = sign;
        } else {
            std::uint32_t normalized = mantissa;
            std::uint32_t exp = 127 - 14;
            while ((normalized & 0x0400u) == 0) {
                normalized <<= 1;
                --exp;
            }
            result = sign | (exp << 23) | ((normalized & 0x03ffu) << 13);
        }
    } else if (exponent == 0x1fu) {
        result = sign | 0x7f800000u | (mantissa << 13);
    } else {
        result = sign | ((exponent + 112) << 23) | (mantissa << 13);
    }
    float value = 0.0f;
    std::memcpy(&value, &result, sizeof(value));
    return value;
}

const chapter01::GgufTensorInfo* find_tensor(
    const chapter01::GgufModelIndex& index,
    const std::string& name) {
    for (const auto& tensor : index.tensors) {
        if (tensor.name == name) return &tensor;
    }
    return nullptr;
}

bool load_named(const std::string& path,
                const chapter01::GgufModelIndex& index,
                const std::string& name,
                chapter01::Tensor* output,
                std::string* error) {
    const auto* info = find_tensor(index, name);
    if (info == nullptr) {
        *error = "missing tensor: " + name;
        return false;
    }
    return chapter01::GgufTensorLoader::load_tensor(path, index, *info, output, error);
}

bool check_vector(const chapter01::Tensor& tensor, std::size_t size, std::string* error) {
    if (tensor.shape().size() != 1 || tensor.numel() != size || tensor.bytes() < size * 2) {
        *error = "unexpected vector shape: " + tensor.name();
        return false;
    }
    return true;
}

std::vector<float> load_floating_vector(const chapter01::Tensor& tensor) {
    std::vector<float> result(tensor.numel(), 0.0f);
    if (tensor.dtype() == chapter01::DType::Float32) {
        std::memcpy(result.data(), tensor.raw_data(), result.size() * sizeof(float));
    } else if (tensor.dtype() == chapter01::DType::Float16) {
        const auto* values = reinterpret_cast<const std::uint16_t*>(tensor.raw_data());
        for (std::size_t i = 0; i < result.size(); ++i) result[i] = half_to_float(values[i]);
    }
    return result;
}

bool q4_linear(const chapter01::Tensor& weight,
               const std::vector<float>& input,
               std::vector<float>* output,
               std::string* error) {
    const auto& shape = weight.shape();
    if (shape.size() != 2 || weight.dtype() != chapter01::DType::Q4_K) {
        *error = "expected 2D Q4_K weight: " + weight.name();
        return false;
    }
    return chapter05_quantization::Q4KLinearFused::forward(
        reinterpret_cast<const std::uint8_t*>(weight.raw_data()), weight.bytes(),
        static_cast<std::size_t>(shape[1]), static_cast<std::size_t>(shape[0]),
        input, output, error);
}

bool q6_linear(const chapter01::Tensor& weight,
               const std::vector<float>& input,
               std::vector<float>* output,
               std::string* error) {
    const auto& shape = weight.shape();
    if (shape.size() != 2 || weight.dtype() != chapter01::DType::Q6_K) {
        *error = "expected 2D Q6_K weight: " + weight.name();
        return false;
    }
    return chapter05_quantization::Q6KLinearFused::forward(
        reinterpret_cast<const std::uint8_t*>(weight.raw_data()), weight.bytes(),
        static_cast<std::size_t>(shape[1]), static_cast<std::size_t>(shape[0]),
        input, output, error);
}

std::vector<float> rms_norm(const std::vector<float>& input,
                            const std::vector<float>& weight,
                            float epsilon) {
    float mean_square = 0.0f;
    for (const float value : input) mean_square += value * value;
    mean_square /= static_cast<float>(input.size());
    const float inverse = 1.0f / std::sqrt(mean_square + epsilon);
    std::vector<float> output(input.size(), 0.0f);
    for (std::size_t i = 0; i < input.size(); ++i) output[i] = input[i] * inverse * weight[i];
    return output;
}

float silu(float value) {
    return value / (1.0f + std::exp(-value));
}

bool finite_vector(const std::vector<float>& values) {
    return std::all_of(values.begin(), values.end(), [](float value) { return std::isfinite(value); });
}

float max_abs(const std::vector<float>& values) {
    float result = 0.0f;
    for (const float value : values) result = std::max(result, std::abs(value));
    return result;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "usage: gguf_first_layer_forward <model.gguf>\n";
        return 2;
    }
    const std::string model_path = argv[1];
    std::string error;
    chapter01::GgufModelIndex index;
    if (!chapter01::GgufReader::read_index(model_path, &index, &error)) {
        std::cerr << "index load failed: " << error << '\n';
        return 1;
    }
    chapter01::ModelMetadata metadata;
    if (!chapter01::ModelMetadataAdapter::from_gguf(index.metadata, &metadata, &error)) {
        std::cerr << "metadata load failed: " << error << '\n';
        return 1;
    }
    chapter01::LlamaTensorLayout layout;
    if (!chapter01::ModelSchema::validate_gguf(metadata, index, &layout, &error)) {
        std::cerr << "schema validation failed: " << error << '\n';
        return 1;
    }

    chapter01::Tensor embedding;
    if (!load_named(model_path, index, layout.embedding, &embedding, &error)) {
        std::cerr << "embedding load failed: " << error << '\n';
        return 1;
    }
    chapter01::RealLlamaLayerWeights layer;
    if (!chapter01::RealLlamaLayerLoader::load(model_path, index, layout, 0, &layer, &error)) {
        std::cerr << "layer load failed: " << error << '\n';
        return 1;
    }
    const std::size_t hidden_size = metadata.hidden_size;
    const std::size_t token_id = 15043;  // "Hello" under the TinyLlama reference tokenizer.
    const std::size_t embedding_block_bytes = 144;
    const std::size_t embedding_row_bytes = hidden_size / 256 * embedding_block_bytes;
    const std::size_t embedding_offset = token_id * embedding_row_bytes;
    if (embedding.dtype() != chapter01::DType::Q4_K || embedding_offset + embedding_row_bytes > embedding.bytes()) {
        std::cerr << "embedding layout is not a contiguous Q4_K token table\n";
        return 1;
    }
    std::vector<float> hidden;
    if (!chapter05_quantization::dequantize_q4_k(
            reinterpret_cast<const std::uint8_t*>(embedding.raw_data()) + embedding_offset,
            embedding_row_bytes, hidden_size, &hidden, &error)) {
        std::cerr << "embedding dequant failed: " << error << '\n';
        return 1;
    }

    const auto norm1_weight = load_floating_vector(layer.attn_norm);
    const auto norm2_weight = load_floating_vector(layer.ffn_norm);
    const auto norm1 = rms_norm(hidden, norm1_weight, metadata.rms_norm_epsilon);
    std::vector<float> query;
    std::vector<float> key;
    std::vector<float> value;
    if (!q4_linear(layer.attn_q, norm1, &query, &error) ||
        !q4_linear(layer.attn_k, norm1, &key, &error) ||
        !q6_linear(layer.attn_v, norm1, &value, &error)) {
        std::cerr << "attention projection failed: " << error << '\n';
        return 1;
    }

    constexpr std::size_t head_dim = 64;
    constexpr std::size_t query_heads = 32;
    constexpr std::size_t kv_heads = 4;
    std::vector<float> attention_output(hidden_size, 0.0f);
    for (std::size_t query_head = 0; query_head < query_heads; ++query_head) {
        const std::size_t kv_head = query_head / (query_heads / kv_heads);
        std::copy_n(value.begin() + kv_head * head_dim, head_dim,
                    attention_output.begin() + query_head * head_dim);
    }
    std::vector<float> projected_attention;
    if (!q4_linear(layer.attn_output, attention_output, &projected_attention, &error)) {
        std::cerr << "attention output projection failed: " << error << '\n';
        return 1;
    }
    for (std::size_t i = 0; i < hidden.size(); ++i) hidden[i] += projected_attention[i];

    const auto norm2 = rms_norm(hidden, norm2_weight, metadata.rms_norm_epsilon);
    std::vector<float> gate;
    std::vector<float> up;
    if (!q4_linear(layer.ffn_gate, norm2, &gate, &error) ||
        !q4_linear(layer.ffn_up, norm2, &up, &error)) {
        std::cerr << "FFN up projection failed: " << error << '\n';
        return 1;
    }
    for (std::size_t i = 0; i < gate.size(); ++i) gate[i] = silu(gate[i]) * up[i];
    std::vector<float> projected_ffn;
    if (!q6_linear(layer.ffn_down, gate, &projected_ffn, &error)) {
        std::cerr << "FFN down projection failed: " << error << '\n';
        return 1;
    }
    for (std::size_t i = 0; i < hidden.size(); ++i) hidden[i] += projected_ffn[i];

    std::cout << "token_id: " << token_id << '\n'
              << "hidden_size: " << hidden.size() << '\n'
              << "query_size: " << query.size() << '\n'
              << "kv_size: " << key.size() << '\n'
              << "output_max_abs: " << max_abs(hidden) << '\n'
              << "output_finite: " << std::boolalpha << finite_vector(hidden) << '\n'
              << "first_layer_forward: " << (finite_vector(hidden) ? "ready" : "failed") << '\n';
    return finite_vector(hidden) ? 0 : 1;
}
