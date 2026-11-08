// Chapter 04/05：真实 TinyLlama 首层多 token CPU forward。
// 职责：验证 RoPE、causal attention score、GQA 和增量 KV Cache 的连接。
// 限制：只执行 layer-0；这是结构 smoke，不代表完整模型或 llama.cpp 数值等价。
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
    const chapter01::GgufModelIndex& index, const std::string& name) {
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

bool linear(const chapter01::Tensor& weight,
            const std::vector<float>& input,
            std::vector<float>* output,
            std::string* error) {
    const auto& shape = weight.shape();
    if (shape.size() != 2) {
        *error = "expected 2D weight: " + weight.name();
        return false;
    }
    const auto rows = static_cast<std::size_t>(shape[0]);
    const auto cols = static_cast<std::size_t>(shape[1]);
    if (weight.dtype() == chapter01::DType::Q4_K) {
        return chapter05_quantization::Q4KLinearFused::forward(
            reinterpret_cast<const std::uint8_t*>(weight.raw_data()), weight.bytes(),
            cols, rows, input, output, error);
    }
    if (weight.dtype() == chapter01::DType::Q6_K) {
        return chapter05_quantization::Q6KLinearFused::forward(
            reinterpret_cast<const std::uint8_t*>(weight.raw_data()), weight.bytes(),
            cols, rows, input, output, error);
    }
    *error = "unsupported linear dtype: " + weight.name();
    return false;
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

void apply_rope(std::vector<float>* values,
                std::size_t head_count,
                std::size_t head_dim,
                std::size_t position,
                std::size_t rope_dim,
                float freq_base) {
    const std::size_t applied_dim = std::min(head_dim, rope_dim);
    for (std::size_t head = 0; head < head_count; ++head) {
        float* vector = values->data() + head * head_dim;
        for (std::size_t pair = 0; pair < applied_dim / 2; ++pair) {
            const float exponent = -2.0f * static_cast<float>(pair) /
                                   static_cast<float>(applied_dim);
            const float angle = static_cast<float>(position) * std::pow(freq_base, exponent);
            const float cosine = std::cos(angle);
            const float sine = std::sin(angle);
            const float x = vector[2 * pair];
            const float y = vector[2 * pair + 1];
            vector[2 * pair] = x * cosine - y * sine;
            vector[2 * pair + 1] = x * sine + y * cosine;
        }
    }
}

float dot(const float* left, const float* right, std::size_t size) {
    float result = 0.0f;
    for (std::size_t i = 0; i < size; ++i) result += left[i] * right[i];
    return result;
}

void softmax(std::vector<float>* values) {
    const float maximum = *std::max_element(values->begin(), values->end());
    float total = 0.0f;
    for (float& value : *values) {
        value = std::exp(value - maximum);
        total += value;
    }
    for (float& value : *values) value /= total;
}

float silu(float value) {
    return value / (1.0f + std::exp(-value));
}

bool finite_vector(const std::vector<float>& values) {
    return std::all_of(values.begin(), values.end(), [](float value) { return std::isfinite(value); });
}

float min_value(const std::vector<float>& values) {
    return *std::min_element(values.begin(), values.end());
}

float max_value(const std::vector<float>& values) {
    return *std::max_element(values.begin(), values.end());
}

bool dequantize_embedding(const chapter01::Tensor& embedding,
                          std::size_t token_id,
                          std::size_t hidden_size,
                          std::vector<float>* output,
                          std::string* error) {
    constexpr std::size_t q4k_block_bytes = 144;
    const std::size_t row_bytes = hidden_size / 256 * q4k_block_bytes;
    const std::size_t offset = token_id * row_bytes;
    if (embedding.dtype() != chapter01::DType::Q4_K ||
        offset + row_bytes > embedding.bytes()) {
        *error = "embedding layout is not a contiguous Q4_K token table";
        return false;
    }
    return chapter05_quantization::dequantize_q4_k(
        reinterpret_cast<const std::uint8_t*>(embedding.raw_data()) + offset,
        row_bytes, hidden_size, output, error);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "usage: gguf_first_layer_multitoken <model.gguf>\n";
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

    constexpr std::size_t head_dim = 64;
    constexpr std::size_t query_heads = 32;
    constexpr std::size_t kv_heads = 4;
    const std::vector<std::size_t> token_ids = {15043, 18439, 1460, 29931, 5288, 29874};
    const std::size_t rope_dim = metadata.rope_dimension_count == 0
        ? head_dim : metadata.rope_dimension_count;
    const auto norm1_weight = load_floating_vector(layer.attn_norm);
    const auto norm2_weight = load_floating_vector(layer.ffn_norm);
    std::vector<float> key_cache;
    std::vector<float> value_cache;
    key_cache.reserve(token_ids.size() * kv_heads * head_dim);
    value_cache.reserve(token_ids.size() * kv_heads * head_dim);
    std::vector<float> all_scores;
    std::vector<float> hidden;

    for (std::size_t position = 0; position < token_ids.size(); ++position) {
        if (!dequantize_embedding(embedding, token_ids[position], metadata.hidden_size,
                                  &hidden, &error)) {
            std::cerr << "embedding dequant failed: " << error << '\n';
            return 1;
        }
        const auto norm1 = rms_norm(hidden, norm1_weight, metadata.rms_norm_epsilon);
        std::vector<float> query;
        std::vector<float> key;
        std::vector<float> value;
        if (!linear(layer.attn_q, norm1, &query, &error) ||
            !linear(layer.attn_k, norm1, &key, &error) ||
            !linear(layer.attn_v, norm1, &value, &error)) {
            std::cerr << "attention projection failed: " << error << '\n';
            return 1;
        }
        apply_rope(&query, query_heads, head_dim, position, rope_dim, metadata.rope_freq_base);
        apply_rope(&key, kv_heads, head_dim, position, rope_dim, metadata.rope_freq_base);
        key_cache.insert(key_cache.end(), key.begin(), key.end());
        value_cache.insert(value_cache.end(), value.begin(), value.end());

        std::vector<float> attention_output(metadata.hidden_size, 0.0f);
        const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
        for (std::size_t query_head = 0; query_head < query_heads; ++query_head) {
            const std::size_t kv_head = query_head / (query_heads / kv_heads);
            std::vector<float> scores(position + 1, 0.0f);
            for (std::size_t key_position = 0; key_position <= position; ++key_position) {
                const float* cached_key = key_cache.data() +
                    (key_position * kv_heads + kv_head) * head_dim;
                scores[key_position] = dot(query.data() + query_head * head_dim,
                                           cached_key, head_dim) * scale;
            }
            softmax(&scores);
            all_scores.insert(all_scores.end(), scores.begin(), scores.end());
            float* output = attention_output.data() + query_head * head_dim;
            for (std::size_t key_position = 0; key_position <= position; ++key_position) {
                const float* cached_value = value_cache.data() +
                    (key_position * kv_heads + kv_head) * head_dim;
                for (std::size_t dimension = 0; dimension < head_dim; ++dimension) {
                    output[dimension] += scores[key_position] * cached_value[dimension];
                }
            }
        }

        std::vector<float> projected_attention;
        if (!linear(layer.attn_output, attention_output, &projected_attention, &error)) {
            std::cerr << "attention output projection failed: " << error << '\n';
            return 1;
        }
        for (std::size_t i = 0; i < hidden.size(); ++i) hidden[i] += projected_attention[i];

        const auto norm2 = rms_norm(hidden, norm2_weight, metadata.rms_norm_epsilon);
        std::vector<float> gate;
        std::vector<float> up;
        if (!linear(layer.ffn_gate, norm2, &gate, &error) ||
            !linear(layer.ffn_up, norm2, &up, &error)) {
            std::cerr << "FFN up projection failed: " << error << '\n';
            return 1;
        }
        for (std::size_t i = 0; i < gate.size(); ++i) gate[i] = silu(gate[i]) * up[i];
        std::vector<float> projected_ffn;
        if (!linear(layer.ffn_down, gate, &projected_ffn, &error)) {
            std::cerr << "FFN down projection failed: " << error << '\n';
            return 1;
        }
        for (std::size_t i = 0; i < hidden.size(); ++i) hidden[i] += projected_ffn[i];
        if (!finite_vector(hidden)) {
            std::cerr << "non-finite hidden state at position " << position << '\n';
            return 1;
        }
    }

    std::cout << "token_count: " << token_ids.size() << '\n'
              << "rope_dim: " << rope_dim << '\n'
              << "rope_freq_base: " << metadata.rope_freq_base << '\n'
              << "kv_cache_tokens: " << key_cache.size() / (kv_heads * head_dim) << '\n'
              << "attention_score_count: " << all_scores.size() << '\n'
              << "attention_score_min: " << min_value(all_scores) << '\n'
              << "attention_score_max: " << max_value(all_scores) << '\n'
              << "last_output_finite: " << std::boolalpha << finite_vector(hidden) << '\n'
              << "first_layer_multitoken: " << (finite_vector(hidden) ? "ready" : "failed") << '\n';
    return finite_vector(hidden) ? 0 : 1;
}
