// 真实模型执行层：实现 Llama Decoder、请求级 KV Cache 与 LM Head。
// 职责：共享量化权重并执行 RMSNorm、RoPE、GQA、SwiGLU 和 LM Head。
// 契约：每个 Context 独占 KV Cache；Model 不随 prefill/decode 改变。
#include "cpp_llm_engine/model/real_llama_cpu_decoder.h"

#include "gguf_reader.h"
#include "gguf_tensor_loader.h"
#include "model_metadata_adapter.h"
#include "model_schema.h"
#include "real_layer_weights.h"

#include "chapter05_quantization/q4k_dequantize.h"
#include "chapter05_quantization/q4k_linear.h"
#include "chapter05_quantization/q6k_linear.h"
#include "chapter05_quantization/q8_0_dequantize.h"
#include "chapter05_quantization/quantized_linear_policy.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <unordered_set>
#include <utility>

namespace chapter01 {
namespace {

using ProfileClock = std::chrono::steady_clock;

class ProfileTimer final {
public:
    explicit ProfileTimer(double* destination)
        : destination_(destination),
          start_(destination == nullptr ? ProfileClock::time_point{} : ProfileClock::now()) {}

    ~ProfileTimer() {
        if (destination_ == nullptr) return;
        *destination_ += std::chrono::duration<double, std::milli>(
            ProfileClock::now() - start_).count();
    }

private:
    double* destination_ = nullptr;
    ProfileClock::time_point start_;
};

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

const GgufTensorInfo* find_tensor(const GgufModelIndex& index, const std::string& name) {
    for (const auto& tensor : index.tensors) {
        if (tensor.name == name) return &tensor;
    }
    return nullptr;
}

bool load_named(const std::string& path,
                const GgufModelIndex& index,
                const std::string& name,
                Tensor* output,
                std::string* error) {
    const auto* info = find_tensor(index, name);
    if (info == nullptr) {
        if (error != nullptr) *error = "missing tensor: " + name;
        return false;
    }
    return GgufTensorLoader::load_tensor(path, index, *info, output, error);
}

std::vector<float> floating_vector(const Tensor& tensor) {
    std::vector<float> result(tensor.numel(), 0.0f);
    if (tensor.dtype() == DType::Float32) {
        std::memcpy(result.data(), tensor.raw_data(), result.size() * sizeof(float));
    } else if (tensor.dtype() == DType::Float16) {
        const auto* values = reinterpret_cast<const std::uint16_t*>(tensor.raw_data());
        for (std::size_t i = 0; i < result.size(); ++i) result[i] = half_to_float(values[i]);
    }
    return result;
}

const chapter05_quantization::QuantizedLinearBackendRegistry&
available_quantized_linear_backends() {
    static const auto registry = [] {
        chapter05_quantization::QuantizedLinearBackendRegistry value;
#if defined(CHAPTER01_HAS_Q4K_AVX2)
        if (chapter05_quantization::q4k_avx2_supported()) {
            value.q4k_avx2 = &chapter05_quantization::q4k_avx2_backend();
        }
#endif
#if defined(CHAPTER01_HAS_Q6K_AVX2)
        if (chapter05_quantization::q6k_avx2_supported()) {
            value.q6k_avx2 = &chapter05_quantization::q6k_avx2_backend();
        }
#endif
        return value;
    }();
    return registry;
}

bool linear(const Tensor& weight,
            const std::vector<float>& input,
            std::vector<float>* output,
            std::string* error,
            const chapter05_quantization::QuantizedLinearPolicy& policy) {
    if (output == nullptr) {
        if (error != nullptr) *error = "linear output must not be null";
        return false;
    }
    const auto& shape = weight.shape();
    if (shape.size() != 2) {
        if (error != nullptr) *error = "expected 2D weight: " + weight.name();
        return false;
    }
    const auto cols = static_cast<std::size_t>(shape[0]);
    const auto rows = static_cast<std::size_t>(shape[1]);
    if (input.size() > std::numeric_limits<std::size_t>::max() / cols ||
        input.size() > std::numeric_limits<std::size_t>::max() / rows) {
        if (error != nullptr) *error = "Linear batch shape overflows host size";
        return false;
    }
    const auto* bytes = reinterpret_cast<const std::uint8_t*>(weight.raw_data());
    cpp_llm_engine::core::WeightFormat format = cpp_llm_engine::core::WeightFormat::Unknown;
    if (weight.dtype() == DType::Q4_K) format = cpp_llm_engine::core::WeightFormat::Q4K;
    if (weight.dtype() == DType::Q3_K) format = cpp_llm_engine::core::WeightFormat::Q3K;
    if (weight.dtype() == DType::Q5_0) format = cpp_llm_engine::core::WeightFormat::Q5_0;
    if (weight.dtype() == DType::Q5_1) format = cpp_llm_engine::core::WeightFormat::Q5_1;
    if (weight.dtype() == DType::Q5_K) format = cpp_llm_engine::core::WeightFormat::Q5K;
    if (weight.dtype() == DType::Q6_K) format = cpp_llm_engine::core::WeightFormat::Q6K;
    if (weight.dtype() == DType::Q8_0) format = cpp_llm_engine::core::WeightFormat::Q8_0;
    if (format == cpp_llm_engine::core::WeightFormat::Unknown) {
        if (error != nullptr) *error = "unsupported linear dtype: " + weight.name();
        return false;
    }
    output->assign(rows, 0.0f);
    const cpp_llm_engine::core::LinearRequest request{
        bytes, weight.bytes(), input.data(), input.size(), output->data(), output->size(),
        rows, cols, format};
    const auto status = chapter05_quantization::compute_quantized_linear(
        request, policy, available_quantized_linear_backends());
    if (!status.is_ok()) {
        output->clear();
        if (error != nullptr) *error = status.message();
        return false;
    }
    if (error != nullptr) error->clear();
    return true;
}

bool same_policy(const chapter05_quantization::QuantizedLinearPolicy& left,
                 const chapter05_quantization::QuantizedLinearPolicy& right) {
    return left.q4k == right.q4k && left.q6k == right.q6k &&
        left.q8_0 == right.q8_0 && left.q3k == right.q3k &&
        left.q5_0 == right.q5_0 &&
        left.q5_1 == right.q5_1 && left.q5k == right.q5k;
}

bool linear_batch(
    const Tensor& weight,
    const std::vector<std::vector<float>>& input,
    std::vector<std::vector<float>>* output,
    std::string* error,
    const chapter05_quantization::QuantizedLinearPolicy& policy,
    RealLlamaDecoderProfile* profile) {
    const auto pack_start = ProfileClock::now();
    if (input.empty() || output == nullptr) {
        if (error != nullptr) *error = "Linear batch input and output must not be empty";
        return false;
    }
    const auto& shape = weight.shape();
    if (shape.size() != 2) {
        if (error != nullptr) *error = "expected 2D batch weight: " + weight.name();
        return false;
    }
    const auto cols = static_cast<std::size_t>(shape[0]);
    const auto rows = static_cast<std::size_t>(shape[1]);
    std::vector<float> flat_input;
    flat_input.reserve(input.size() * cols);
    for (const auto& row : input) {
        if (row.size() != cols) {
            if (error != nullptr) *error = "Linear batch input width does not match weight";
            return false;
        }
        flat_input.insert(flat_input.end(), row.begin(), row.end());
    }
    if (profile != nullptr) {
        profile->batch_input_pack_ms += std::chrono::duration<double, std::milli>(
            ProfileClock::now() - pack_start).count();
    }
    cpp_llm_engine::core::WeightFormat format = cpp_llm_engine::core::WeightFormat::Unknown;
    if (weight.dtype() == DType::Q4_K) format = cpp_llm_engine::core::WeightFormat::Q4K;
    if (weight.dtype() == DType::Q3_K) format = cpp_llm_engine::core::WeightFormat::Q3K;
    if (weight.dtype() == DType::Q5_0) format = cpp_llm_engine::core::WeightFormat::Q5_0;
    if (weight.dtype() == DType::Q5_1) format = cpp_llm_engine::core::WeightFormat::Q5_1;
    if (weight.dtype() == DType::Q5_K) format = cpp_llm_engine::core::WeightFormat::Q5K;
    if (weight.dtype() == DType::Q6_K) format = cpp_llm_engine::core::WeightFormat::Q6K;
    if (weight.dtype() == DType::Q8_0) format = cpp_llm_engine::core::WeightFormat::Q8_0;
    if (format == cpp_llm_engine::core::WeightFormat::Unknown) {
        if (error != nullptr) *error = "unsupported batch Linear dtype: " + weight.name();
        return false;
    }

    std::vector<float> flat_output(input.size() * rows, 0.0f);
    const cpp_llm_engine::core::LinearBatchRequest request{
        weight.raw_data(), weight.bytes(), flat_input.data(), flat_input.size(),
        flat_output.data(), flat_output.size(), input.size(), rows, cols, format};
    const auto kernel_start = ProfileClock::now();
    const auto status = chapter05_quantization::compute_quantized_linear_batch(
        request, policy, available_quantized_linear_backends());
    if (profile != nullptr) {
        profile->batch_kernel_ms += std::chrono::duration<double, std::milli>(
            ProfileClock::now() - kernel_start).count();
        ++profile->batch_linear_calls;
    }
    if (!status.is_ok()) {
        if (error != nullptr) *error = status.message();
        return false;
    }
    const auto unpack_start = ProfileClock::now();
    output->assign(input.size(), std::vector<float>(rows, 0.0f));
    for (std::size_t batch = 0; batch < input.size(); ++batch) {
        std::copy_n(flat_output.data() + batch * rows, rows, (*output)[batch].data());
    }
    if (profile != nullptr) {
        profile->batch_output_unpack_ms += std::chrono::duration<double, std::milli>(
            ProfileClock::now() - unpack_start).count();
    }
    if (error != nullptr) error->clear();
    return true;
}

std::vector<float> rms_norm(const std::vector<float>& input,
                            const std::vector<float>& weight,
                            float epsilon) {
    float mean_square = 0.0f;
    for (float value : input) mean_square += value * value;
    mean_square /= static_cast<float>(input.size());
    const float inverse = 1.0f / std::sqrt(mean_square + epsilon);
    std::vector<float> output(input.size(), 0.0f);
    for (std::size_t i = 0; i < input.size(); ++i) output[i] = input[i] * inverse * weight[i];
    return output;
}

bool rms_norm_heads(std::vector<float>* values,
                    std::size_t head_count,
                    std::size_t head_dimension,
                    const std::vector<float>& weight,
                    float epsilon,
                    std::string* error) {
    if (values == nullptr || head_count == 0 || head_dimension == 0 ||
        values->size() != head_count * head_dimension ||
        weight.size() != head_dimension) {
        if (error != nullptr) *error = "per-head RMSNorm dimensions are invalid";
        return false;
    }
    for (std::size_t head = 0; head < head_count; ++head) {
        float mean_square = 0.0f;
        float* vector = values->data() + head * head_dimension;
        for (std::size_t i = 0; i < head_dimension; ++i) {
            mean_square += vector[i] * vector[i];
        }
        mean_square /= static_cast<float>(head_dimension);
        const float inverse = 1.0f / std::sqrt(mean_square + epsilon);
        for (std::size_t i = 0; i < head_dimension; ++i) {
            vector[i] *= inverse * weight[i];
        }
    }
    return true;
}

void apply_rope(std::vector<float>* values,
                std::size_t head_count,
                std::size_t head_dim,
                std::size_t position,
                std::size_t rope_dim,
                float freq_base,
                bool split_half_pairs) {
    const std::size_t applied_dim = std::min(head_dim, rope_dim);
    for (std::size_t head = 0; head < head_count; ++head) {
        float* vector = values->data() + head * head_dim;
        for (std::size_t pair = 0; pair < applied_dim / 2; ++pair) {
            const float exponent = -2.0f * static_cast<float>(pair) /
                                   static_cast<float>(applied_dim);
            const float angle = static_cast<float>(position) * std::pow(freq_base, exponent);
            const float cosine = std::cos(angle);
            const float sine = std::sin(angle);
            const std::size_t first = split_half_pairs ? pair : 2 * pair;
            const std::size_t second = split_half_pairs
                ? pair + applied_dim / 2 : 2 * pair + 1;
            const float x = vector[first];
            const float y = vector[second];
            vector[first] = x * cosine - y * sine;
            vector[second] = x * sine + y * cosine;
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

struct ModelLayer {
    RealLlamaLayerWeights weights;
    std::vector<float> attention_norm;
    std::vector<float> query_norm;
    std::vector<float> key_norm;
    std::vector<float> ffn_norm;
};

struct ContextLayer {
    std::vector<float> keys;
    std::vector<float> values;
};

struct ModelData {
    ModelMetadata metadata;
    LlamaTensorLayout layout;
    Tensor embedding;
    Tensor output_weight;
    std::vector<float> output_norm;
    std::vector<ModelLayer> layers;
    std::size_t head_dim = 0;
    std::size_t query_heads = 0;
    std::size_t kv_heads = 0;
    std::size_t rope_dim = 0;
    bool rope_uses_split_half_pairs = false;
    std::size_t attention_width = 0;
    std::size_t weight_payload_bytes = 0;
};

std::size_t layer_payload_bytes(const RealLlamaLayerWeights& weights) {
    return weights.attn_norm.bytes() + weights.attn_q.bytes() +
        weights.attn_k.bytes() + weights.attn_v.bytes() +
        weights.attn_output.bytes() + weights.attn_q_norm.bytes() +
        weights.attn_k_norm.bytes() + weights.ffn_norm.bytes() +
        weights.ffn_gate.bytes() + weights.ffn_up.bytes() + weights.ffn_down.bytes();
}

bool decode_token_embedding(const ModelData& model,
                            TokenId token,
                            std::vector<float>* output,
                            std::string* error) {
    if (output == nullptr) {
        if (error != nullptr) *error = "token embedding output must not be null";
        return false;
    }
    if (token < 0 || static_cast<std::size_t>(token) >= model.metadata.vocab_size) {
        if (error != nullptr) *error = "token is outside embedding table";
        return false;
    }
    std::size_t block_elements = 0;
    std::size_t block_bytes = 0;
    if (model.embedding.dtype() == DType::Q4_K) {
        block_elements = chapter05_quantization::kQ4KBlockElements;
        block_bytes = chapter05_quantization::kQ4KBlockBytes;
    } else if (model.embedding.dtype() == DType::Q8_0) {
        block_elements = chapter05_quantization::kQ8_0BlockElements;
        block_bytes = chapter05_quantization::kQ8_0BlockBytes;
    } else {
        if (error != nullptr) *error = "embedding table must use Q4_K or Q8_0";
        return false;
    }
    if (model.metadata.hidden_size % block_elements != 0) {
        if (error != nullptr) *error = "embedding width is not quantization-block aligned";
        return false;
    }
    const std::size_t row_bytes = model.metadata.hidden_size / block_elements * block_bytes;
    const std::size_t offset = static_cast<std::size_t>(token) * row_bytes;
    if (offset > model.embedding.bytes() || row_bytes > model.embedding.bytes() - offset) {
        if (error != nullptr) *error = "token embedding payload exceeds table";
        return false;
    }
    const auto* row =
        reinterpret_cast<const std::uint8_t*>(model.embedding.raw_data()) + offset;
    return model.embedding.dtype() == DType::Q4_K
        ? chapter05_quantization::dequantize_q4_k(
              row, row_bytes, model.metadata.hidden_size, output, error)
        : chapter05_quantization::dequantize_q8_0(
              row, row_bytes, model.metadata.hidden_size, output, error);
}

}  // namespace

struct RealLlamaCpuModel::Impl {
    std::shared_ptr<const ModelData> data;
};

struct RealLlamaCpuContext::Impl {
    std::shared_ptr<const ModelData> model;
    std::vector<ContextLayer> layers;
    std::shared_ptr<RealLlamaKvDataPlane> external_kv;
    std::size_t position = 0;
    std::size_t attention_score_count = 0;
    bool profiling_enabled = false;
    chapter05_quantization::QuantizedLinearPolicy attention_linear_policy;
    chapter05_quantization::QuantizedLinearPolicy ffn_linear_policy;
    chapter05_quantization::QuantizedLinearPolicy lm_head_linear_policy;
    RealLlamaDecoderProfile profile;

    bool forward_token(TokenId token, std::vector<float>* logits, std::string* error) {
        ProfileTimer total_timer(profiling_enabled ? &profile.total_ms : nullptr);
        std::vector<float> hidden;
        {
            ProfileTimer timer(profiling_enabled ? &profile.embedding_ms : nullptr);
            if (!decode_token_embedding(*model, token, &hidden, error)) return false;
        }

        return forward_hidden(std::move(hidden), logits, nullptr, error);
    }

    bool forward_embedding(const float* embedding,
                           std::size_t hidden_size,
                           std::vector<float>* logits,
                           std::string* error) {
        ProfileTimer total_timer(profiling_enabled ? &profile.total_ms : nullptr);
        if (embedding == nullptr || hidden_size != model->metadata.hidden_size) {
            if (error != nullptr) *error = "input embedding width does not match decoder hidden size";
            return false;
        }
        return forward_hidden(
            std::vector<float>(embedding, embedding + hidden_size),
            logits, nullptr, error);
    }

    bool forward_layer(
        std::vector<float>* hidden,
        std::size_t layer_index,
        std::string* error) {
        if (hidden == nullptr || layer_index >= model->layers.size()) {
            if (error != nullptr) *error = "decoder layer input is invalid";
            return false;
        }
        const auto& weights = model->layers[layer_index];
        auto& cache = layers[layer_index];
        std::vector<float> normalized;
        {
            ProfileTimer timer(profiling_enabled ? &profile.normalization_ms : nullptr);
            normalized = rms_norm(*hidden, weights.attention_norm,
                                  model->metadata.rms_norm_epsilon);
        }
        std::vector<float> query;
        std::vector<float> key;
        std::vector<float> value;
        {
            ProfileTimer timer(profiling_enabled ? &profile.qkv_projection_ms : nullptr);
            if (!linear(weights.weights.attn_q, normalized, &query, error,
                        attention_linear_policy) ||
                !linear(weights.weights.attn_k, normalized, &key, error,
                        attention_linear_policy) ||
                !linear(weights.weights.attn_v, normalized, &value, error,
                        attention_linear_policy)) {
                return false;
            }
        }
        {
            ProfileTimer timer(profiling_enabled ? &profile.rope_ms : nullptr);
            if ((!weights.query_norm.empty() &&
                 !rms_norm_heads(&query, model->query_heads, model->head_dim,
                                 weights.query_norm, model->metadata.rms_norm_epsilon,
                                 error)) ||
                (!weights.key_norm.empty() &&
                 !rms_norm_heads(&key, model->kv_heads, model->head_dim,
                                 weights.key_norm, model->metadata.rms_norm_epsilon,
                                 error))) {
                return false;
            }
            apply_rope(&query, model->query_heads, model->head_dim, position,
                       model->rope_dim, model->metadata.rope_freq_base,
                       model->rope_uses_split_half_pairs);
            apply_rope(&key, model->kv_heads, model->head_dim, position,
                       model->rope_dim, model->metadata.rope_freq_base,
                       model->rope_uses_split_half_pairs);
        }
        std::vector<float> attention_output(model->attention_width, 0.0f);
        {
            ProfileTimer timer(profiling_enabled ? &profile.attention_ms : nullptr);
            if (external_kv != nullptr) {
                if (!external_kv->write_token(
                        layer_index, position, key.data(), value.data(), key.size(), error)) {
                    return false;
                }
            } else {
                cache.keys.insert(cache.keys.end(), key.begin(), key.end());
                cache.values.insert(cache.values.end(), value.begin(), value.end());
            }
            const float scale = 1.0f / std::sqrt(static_cast<float>(model->head_dim));
            for (std::size_t query_head = 0; query_head < model->query_heads; ++query_head) {
                const std::size_t kv_head = query_head /
                    (model->query_heads / model->kv_heads);
                std::vector<float> scores(position + 1, 0.0f);
                for (std::size_t key_position = 0; key_position <= position; ++key_position) {
                    const float* cached_key = external_kv != nullptr
                        ? external_kv->key_data(layer_index, key_position, kv_head)
                        : cache.keys.data() +
                            (key_position * model->kv_heads + kv_head) * model->head_dim;
                    scores[key_position] = dot(
                        query.data() + query_head * model->head_dim,
                        cached_key, model->head_dim) * scale;
                }
                attention_score_count += scores.size();
                softmax(&scores);
                float* output = attention_output.data() + query_head * model->head_dim;
                for (std::size_t key_position = 0; key_position <= position; ++key_position) {
                    const float* cached_value = external_kv != nullptr
                        ? external_kv->value_data(layer_index, key_position, kv_head)
                        : cache.values.data() +
                            (key_position * model->kv_heads + kv_head) * model->head_dim;
                    for (std::size_t dimension = 0; dimension < model->head_dim; ++dimension) {
                        output[dimension] += scores[key_position] * cached_value[dimension];
                    }
                }
            }
        }
        std::vector<float> attention_projection;
        {
            ProfileTimer timer(profiling_enabled ? &profile.attention_output_ms : nullptr);
            if (!linear(weights.weights.attn_output, attention_output,
                        &attention_projection, error, attention_linear_policy)) return false;
            for (std::size_t i = 0; i < hidden->size(); ++i) {
                (*hidden)[i] += attention_projection[i];
            }
        }
        std::vector<float> ffn_input;
        {
            ProfileTimer timer(profiling_enabled ? &profile.normalization_ms : nullptr);
            ffn_input = rms_norm(*hidden, weights.ffn_norm,
                                 model->metadata.rms_norm_epsilon);
        }
        {
            ProfileTimer timer(profiling_enabled ? &profile.ffn_ms : nullptr);
            std::vector<float> gate;
            std::vector<float> up;
            {
                ProfileTimer sub_timer(profiling_enabled ? &profile.ffn_gate_ms : nullptr);
                if (!linear(weights.weights.ffn_gate, ffn_input, &gate, error,
                            ffn_linear_policy)) return false;
            }
            {
                ProfileTimer sub_timer(profiling_enabled ? &profile.ffn_up_ms : nullptr);
                if (!linear(weights.weights.ffn_up, ffn_input, &up, error,
                            ffn_linear_policy)) return false;
            }
            {
                ProfileTimer sub_timer(
                    profiling_enabled ? &profile.ffn_activation_ms : nullptr);
                for (std::size_t i = 0; i < gate.size(); ++i) {
                    gate[i] = gate[i] / (1.0f + std::exp(-gate[i])) * up[i];
                }
            }
            std::vector<float> ffn_projection;
            {
                ProfileTimer sub_timer(profiling_enabled ? &profile.ffn_down_ms : nullptr);
                if (!linear(weights.weights.ffn_down, gate, &ffn_projection, error,
                            ffn_linear_policy)) return false;
            }
            {
                ProfileTimer sub_timer(profiling_enabled ? &profile.ffn_residual_ms : nullptr);
                for (std::size_t i = 0; i < hidden->size(); ++i) {
                    (*hidden)[i] += ffn_projection[i];
                }
            }
        }
        return true;
    }

    bool finish_hidden(
        std::vector<float> hidden,
        std::vector<float>* logits,
        std::string* error) {
        if (logits != nullptr) {
            std::vector<float> final_hidden;
            {
                ProfileTimer timer(profiling_enabled ? &profile.normalization_ms : nullptr);
                final_hidden = rms_norm(hidden, model->output_norm,
                                        model->metadata.rms_norm_epsilon);
            }
            {
                ProfileTimer timer(profiling_enabled ? &profile.lm_head_ms : nullptr);
                if (!linear(model->output_weight, final_hidden, logits, error,
                            lm_head_linear_policy)) return false;
            }
            if (logits->size() != model->metadata.vocab_size ||
                !std::all_of(logits->begin(), logits->end(),
                             [](float value) { return std::isfinite(value); })) {
                if (error != nullptr) *error = "decoder produced invalid logits";
                return false;
            }
        }
        if (external_kv != nullptr && !external_kv->commit(position + 1, error)) {
            return false;
        }
        ++position;
        if (profiling_enabled) ++profile.forward_tokens;
        return true;
    }

    bool forward_hidden(std::vector<float> hidden,
                        std::vector<float>* logits,
                        std::vector<float>* decoder_hidden,
                        std::string* error) {
        const std::size_t initial_position = position;
        if (external_kv != nullptr &&
            !external_kv->reserve(initial_position + 1, error)) {
            return false;
        }
        for (std::size_t layer_index = 0; layer_index < model->layers.size(); ++layer_index) {
            if (!forward_layer(&hidden, layer_index, error)) {
                if (external_kv != nullptr) external_kv->trim(initial_position);
                return false;
            }
        }
        std::vector<float> hidden_result;
        if (decoder_hidden != nullptr) hidden_result = hidden;
        if (!finish_hidden(std::move(hidden), logits, error)) {
            if (external_kv != nullptr) external_kv->trim(initial_position);
            return false;
        }
        if (decoder_hidden != nullptr) *decoder_hidden = std::move(hidden_result);
        return true;
    }
};

std::unique_ptr<RealLlamaCpuModel> RealLlamaCpuModel::load(
    const std::string& model_path,
    std::size_t layer_count,
    std::string* error) {
    GgufModelIndex index;
    if (!GgufReader::read_index(model_path, &index, error)) return nullptr;
    auto data = std::make_shared<ModelData>();
    if (!ModelMetadataAdapter::from_gguf(index.metadata, &data->metadata, error)) {
        return nullptr;
    }
    if (data->metadata.architecture != "llama" &&
        data->metadata.architecture != "qwen3") {
        if (error != nullptr) {
            *error = "dense CPU model supports llama and qwen3 architectures";
        }
        return nullptr;
    }
    if (!ModelSchema::validate_gguf(data->metadata, index, &data->layout, error)) {
        return nullptr;
    }
    if (layer_count == 0) layer_count = data->metadata.layer_count;
    if (layer_count > data->metadata.layer_count || layer_count == 0) {
        if (error != nullptr) *error = "requested decoder layer count is invalid";
        return nullptr;
    }
    data->query_heads = data->metadata.attention_head_count;
    data->kv_heads = data->metadata.attention_head_count_kv;
    if (data->query_heads == 0 || data->kv_heads == 0 ||
        data->query_heads % data->kv_heads != 0) {
        if (error != nullptr) *error = "attention head metadata is invalid";
        return nullptr;
    }
    if (data->metadata.attention_key_length != 0) {
        if (data->metadata.attention_value_length !=
            data->metadata.attention_key_length) {
            if (error != nullptr) *error = "CPU decoder requires equal key/value head lengths";
            return nullptr;
        }
        data->head_dim = data->metadata.attention_key_length;
    } else {
        if (data->metadata.hidden_size % data->query_heads != 0) {
            if (error != nullptr) *error = "hidden size is not divisible by query heads";
            return nullptr;
        }
        data->head_dim = data->metadata.hidden_size / data->query_heads;
    }
    data->attention_width = data->query_heads * data->head_dim;
    data->rope_dim = data->metadata.rope_dimension_count == 0
        ? data->head_dim : data->metadata.rope_dimension_count;
    data->rope_uses_split_half_pairs = data->metadata.architecture == "qwen3";
    if (!load_named(model_path, index, data->layout.embedding, &data->embedding, error) ||
        !load_named(model_path, index, data->layout.output, &data->output_weight, error)) {
        return nullptr;
    }
    Tensor output_norm;
    if (!load_named(model_path, index, data->layout.output_norm, &output_norm, error)) {
        return nullptr;
    }
    data->weight_payload_bytes = data->embedding.bytes() + data->output_weight.bytes() +
        output_norm.bytes();
    data->output_norm = floating_vector(output_norm);
    data->layers.resize(layer_count);
    for (std::size_t layer_index = 0; layer_index < layer_count; ++layer_index) {
        auto& layer = data->layers[layer_index];
        if (!RealLlamaLayerLoader::load(model_path, index, data->layout, layer_index,
                                        &layer.weights, error)) return nullptr;
        data->weight_payload_bytes += layer_payload_bytes(layer.weights);
        layer.attention_norm = floating_vector(layer.weights.attn_norm);
        if (layer.weights.attn_q_norm.bytes() > 0) {
            layer.query_norm = floating_vector(layer.weights.attn_q_norm);
            layer.key_norm = floating_vector(layer.weights.attn_k_norm);
        }
        layer.ffn_norm = floating_vector(layer.weights.ffn_norm);
    }
    auto impl = std::make_unique<Impl>();
    impl->data = std::move(data);
    if (error != nullptr) error->clear();
    return std::unique_ptr<RealLlamaCpuModel>(new RealLlamaCpuModel(std::move(impl)));
}

RealLlamaCpuModel::RealLlamaCpuModel(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}

RealLlamaCpuModel::~RealLlamaCpuModel() = default;
RealLlamaCpuModel::RealLlamaCpuModel(RealLlamaCpuModel&&) noexcept = default;
RealLlamaCpuModel& RealLlamaCpuModel::operator=(RealLlamaCpuModel&&) noexcept = default;

std::unique_ptr<RealLlamaCpuContext> RealLlamaCpuModel::create_context() const {
    auto context = std::make_unique<RealLlamaCpuContext::Impl>();
    context->model = impl_->data;
    context->layers.resize(impl_->data->layers.size());
    return std::unique_ptr<RealLlamaCpuContext>(
        new RealLlamaCpuContext(std::move(context)));
}

std::size_t RealLlamaCpuModel::layer_count() const { return impl_->data->layers.size(); }
std::size_t RealLlamaCpuModel::model_layer_count() const {
    return impl_->data->metadata.layer_count;
}
const std::string& RealLlamaCpuModel::architecture() const {
    return impl_->data->metadata.architecture;
}
std::size_t RealLlamaCpuModel::vocab_size() const { return impl_->data->metadata.vocab_size; }
TokenId RealLlamaCpuModel::eos_token_id() const { return impl_->data->metadata.eos_token_id; }
std::size_t RealLlamaCpuModel::weight_payload_bytes() const {
    return impl_->data->weight_payload_bytes;
}

cpp_llm_engine::core::DecoderModelContract RealLlamaCpuModel::decoder_contract() const {
    return {
        impl_->data->metadata.vocab_size,
        impl_->data->layers.size(),
        impl_->data->metadata.max_position,
        {
            impl_->data->metadata.hidden_size,
            impl_->data->layout.intermediate_size,
            impl_->data->query_heads,
            impl_->data->kv_heads,
            impl_->data->head_dim,
            impl_->data->rope_dim,
            cpp_llm_engine::core::DecoderNormalizationKind::RmsNorm,
            cpp_llm_engine::core::DecoderActivationKind::SwiGlu,
            cpp_llm_engine::core::DecoderPositionEncodingKind::Rotary,
            true,
            true,
        },
    };
}

bool RealLlamaCpuModel::embed_tokens(
    const TokenSequence& tokens,
    cpp_llm_engine::core::DecoderEmbeddingSequence* embeddings,
    std::string* error) const {
    if (tokens.empty() || embeddings == nullptr) {
        if (error != nullptr) *error = "token embedding requires tokens and output";
        return false;
    }
    cpp_llm_engine::core::DecoderEmbeddingSequence result;
    result.token_count = tokens.size();
    result.hidden_size = impl_->data->metadata.hidden_size;
    if (tokens.size() > std::numeric_limits<std::size_t>::max() / result.hidden_size) {
        if (error != nullptr) *error = "token embedding output shape overflows";
        return false;
    }
    result.values.reserve(tokens.size() * result.hidden_size);
    for (const TokenId token : tokens) {
        std::vector<float> row;
        if (!decode_token_embedding(*impl_->data, token, &row, error)) return false;
        result.values.insert(result.values.end(), row.begin(), row.end());
    }
    *embeddings = std::move(result);
    if (error != nullptr) error->clear();
    return true;
}

bool RealLlamaCpuModel::decode_batch(
    const std::vector<RealLlamaCpuBatchItem>& items,
    std::vector<std::vector<float>>* logits,
    std::string* error) const {
    if (items.empty() || logits == nullptr) {
        if (error != nullptr) *error = "decode batch requires items and logits output";
        return false;
    }

    struct Snapshot {
        RealLlamaCpuContext::Impl* context = nullptr;
        std::size_t position = 0;
        std::size_t attention_score_count = 0;
        RealLlamaDecoderProfile profile;
        std::vector<std::pair<std::size_t, std::size_t>> cache_sizes;
    };

    std::unordered_set<RealLlamaCpuContext*> unique_contexts;
    std::vector<Snapshot> snapshots;
    std::vector<std::vector<float>> hidden(items.size());
    RealLlamaDecoderProfile batch_linear_profile;
    snapshots.reserve(items.size());
    const auto batch_start = ProfileClock::now();

    for (std::size_t index = 0; index < items.size(); ++index) {
        auto* context = items[index].context;
        if (context == nullptr || context->impl_->model != impl_->data ||
            context->impl_->position == 0 || !unique_contexts.insert(context).second) {
            if (error != nullptr) {
                *error = "decode batch requires unique, prefilled contexts from this model";
            }
            return false;
        }
        Snapshot snapshot;
        snapshot.context = context->impl_.get();
        snapshot.position = snapshot.context->position;
        snapshot.attention_score_count = snapshot.context->attention_score_count;
        snapshot.profile = snapshot.context->profile;
        if (snapshot.context->external_kv != nullptr) {
            if (snapshot.context->external_kv->token_count() != snapshot.position ||
                !snapshot.context->external_kv->reserve(snapshot.position + 1, error)) {
                if (error != nullptr && error->empty()) {
                    *error = "external KV state does not match batch context";
                }
                for (auto& previous : snapshots) {
                    if (previous.context->external_kv != nullptr) {
                        previous.context->external_kv->trim(previous.position);
                    }
                }
                return false;
            }
        } else {
            snapshot.cache_sizes.reserve(snapshot.context->layers.size());
            for (const auto& layer : snapshot.context->layers) {
                snapshot.cache_sizes.emplace_back(layer.keys.size(), layer.values.size());
            }
        }
        snapshots.push_back(std::move(snapshot));
    }
    const auto* policy_context = snapshots.front().context;
    for (const auto& snapshot : snapshots) {
        if (!same_policy(snapshot.context->attention_linear_policy,
                         policy_context->attention_linear_policy) ||
            !same_policy(snapshot.context->ffn_linear_policy,
                         policy_context->ffn_linear_policy) ||
            !same_policy(snapshot.context->lm_head_linear_policy,
                         policy_context->lm_head_linear_policy)) {
            if (error != nullptr) *error = "decode batch contexts use different Linear policies";
            return false;
        }
    }

    const auto rollback = [&snapshots]() {
        for (auto& snapshot : snapshots) {
            snapshot.context->position = snapshot.position;
            snapshot.context->attention_score_count = snapshot.attention_score_count;
            snapshot.context->profile = snapshot.profile;
            if (snapshot.context->external_kv != nullptr) {
                snapshot.context->external_kv->trim(snapshot.position);
            } else {
                for (std::size_t layer = 0; layer < snapshot.cache_sizes.size(); ++layer) {
                    snapshot.context->layers[layer].keys.resize(snapshot.cache_sizes[layer].first);
                    snapshot.context->layers[layer].values.resize(snapshot.cache_sizes[layer].second);
                }
            }
        }
    };

    for (std::size_t index = 0; index < items.size(); ++index) {
        bool embedded = false;
        {
            ProfileTimer timer(snapshots[index].context->profiling_enabled
                ? &snapshots[index].context->profile.embedding_ms : nullptr);
            embedded = decode_token_embedding(
                *impl_->data, items[index].token, &hidden[index], error);
        }
        if (!embedded) {
            rollback();
            return false;
        }
    }

    const auto add_profile_time = [&snapshots](
        double RealLlamaDecoderProfile::*field,
        ProfileClock::time_point start) {
        const double elapsed = std::chrono::duration<double, std::milli>(
            ProfileClock::now() - start).count();
        for (const auto& snapshot : snapshots) {
            if (snapshot.context->profiling_enabled) {
                snapshot.context->profile.*field += elapsed;
            }
        }
    };

    for (std::size_t layer = 0; layer < impl_->data->layers.size(); ++layer) {
        const auto& weights = impl_->data->layers[layer];
        auto stage_start = ProfileClock::now();
        std::vector<std::vector<float>> normalized(items.size());
        for (std::size_t index = 0; index < items.size(); ++index) {
            normalized[index] = rms_norm(hidden[index], weights.attention_norm,
                                         impl_->data->metadata.rms_norm_epsilon);
        }
        add_profile_time(&RealLlamaDecoderProfile::normalization_ms, stage_start);

        stage_start = ProfileClock::now();
        std::vector<std::vector<float>> query;
        std::vector<std::vector<float>> key;
        std::vector<std::vector<float>> value;
        if (!linear_batch(weights.weights.attn_q, normalized, &query, error,
                          policy_context->attention_linear_policy, &batch_linear_profile) ||
            !linear_batch(weights.weights.attn_k, normalized, &key, error,
                          policy_context->attention_linear_policy, &batch_linear_profile) ||
            !linear_batch(weights.weights.attn_v, normalized, &value, error,
                          policy_context->attention_linear_policy, &batch_linear_profile)) {
            rollback();
            return false;
        }
        add_profile_time(&RealLlamaDecoderProfile::qkv_projection_ms, stage_start);

        stage_start = ProfileClock::now();
        for (std::size_t index = 0; index < items.size(); ++index) {
            if ((!weights.query_norm.empty() &&
                 !rms_norm_heads(&query[index], impl_->data->query_heads,
                                 impl_->data->head_dim, weights.query_norm,
                                 impl_->data->metadata.rms_norm_epsilon, error)) ||
                (!weights.key_norm.empty() &&
                 !rms_norm_heads(&key[index], impl_->data->kv_heads,
                                 impl_->data->head_dim, weights.key_norm,
                                 impl_->data->metadata.rms_norm_epsilon, error))) {
                rollback();
                return false;
            }
            apply_rope(&query[index], impl_->data->query_heads, impl_->data->head_dim,
                       snapshots[index].context->position, impl_->data->rope_dim,
                       impl_->data->metadata.rope_freq_base,
                       impl_->data->rope_uses_split_half_pairs);
            apply_rope(&key[index], impl_->data->kv_heads, impl_->data->head_dim,
                       snapshots[index].context->position, impl_->data->rope_dim,
                       impl_->data->metadata.rope_freq_base,
                       impl_->data->rope_uses_split_half_pairs);
        }
        add_profile_time(&RealLlamaDecoderProfile::rope_ms, stage_start);

        stage_start = ProfileClock::now();
        std::vector<std::vector<float>> attention_output(
            items.size(), std::vector<float>(impl_->data->attention_width, 0.0f));
        for (std::size_t index = 0; index < items.size(); ++index) {
            auto* context = snapshots[index].context;
            auto& cache = context->layers[layer];
            if (context->external_kv != nullptr) {
                if (!context->external_kv->write_token(
                        layer, context->position, key[index].data(), value[index].data(),
                        key[index].size(), error)) {
                    rollback();
                    return false;
                }
            } else {
                cache.keys.insert(cache.keys.end(), key[index].begin(), key[index].end());
                cache.values.insert(cache.values.end(), value[index].begin(), value[index].end());
            }
            const float scale = 1.0f / std::sqrt(static_cast<float>(impl_->data->head_dim));
            for (std::size_t query_head = 0; query_head < impl_->data->query_heads;
                 ++query_head) {
                const std::size_t kv_head = query_head /
                    (impl_->data->query_heads / impl_->data->kv_heads);
                std::vector<float> scores(context->position + 1, 0.0f);
                for (std::size_t key_position = 0; key_position <= context->position;
                     ++key_position) {
                    const float* cached_key = context->external_kv != nullptr
                        ? context->external_kv->key_data(layer, key_position, kv_head)
                        : cache.keys.data() +
                            (key_position * impl_->data->kv_heads + kv_head) *
                                impl_->data->head_dim;
                    scores[key_position] = dot(
                        query[index].data() + query_head * impl_->data->head_dim,
                        cached_key, impl_->data->head_dim) * scale;
                }
                context->attention_score_count += scores.size();
                softmax(&scores);
                float* output = attention_output[index].data() +
                    query_head * impl_->data->head_dim;
                for (std::size_t key_position = 0; key_position <= context->position;
                     ++key_position) {
                    const float* cached_value = context->external_kv != nullptr
                        ? context->external_kv->value_data(layer, key_position, kv_head)
                        : cache.values.data() +
                            (key_position * impl_->data->kv_heads + kv_head) *
                                impl_->data->head_dim;
                    for (std::size_t dimension = 0; dimension < impl_->data->head_dim;
                         ++dimension) {
                        output[dimension] += scores[key_position] * cached_value[dimension];
                    }
                }
            }
        }
        add_profile_time(&RealLlamaDecoderProfile::attention_ms, stage_start);

        stage_start = ProfileClock::now();
        std::vector<std::vector<float>> attention_projection;
        if (!linear_batch(weights.weights.attn_output, attention_output,
                          &attention_projection, error,
                          policy_context->attention_linear_policy, &batch_linear_profile)) {
            rollback();
            return false;
        }
        for (std::size_t index = 0; index < items.size(); ++index) {
            for (std::size_t dimension = 0; dimension < hidden[index].size(); ++dimension) {
                hidden[index][dimension] += attention_projection[index][dimension];
            }
        }
        add_profile_time(&RealLlamaDecoderProfile::attention_output_ms, stage_start);

        stage_start = ProfileClock::now();
        std::vector<std::vector<float>> ffn_input(items.size());
        for (std::size_t index = 0; index < items.size(); ++index) {
            ffn_input[index] = rms_norm(hidden[index], weights.ffn_norm,
                                        impl_->data->metadata.rms_norm_epsilon);
        }
        add_profile_time(&RealLlamaDecoderProfile::normalization_ms, stage_start);

        const auto ffn_start = ProfileClock::now();
        stage_start = ProfileClock::now();
        std::vector<std::vector<float>> gate;
        if (!linear_batch(weights.weights.ffn_gate, ffn_input, &gate, error,
                          policy_context->ffn_linear_policy, &batch_linear_profile)) {
            rollback();
            return false;
        }
        add_profile_time(&RealLlamaDecoderProfile::ffn_gate_ms, stage_start);

        stage_start = ProfileClock::now();
        std::vector<std::vector<float>> up;
        if (!linear_batch(weights.weights.ffn_up, ffn_input, &up, error,
                          policy_context->ffn_linear_policy, &batch_linear_profile)) {
            rollback();
            return false;
        }
        add_profile_time(&RealLlamaDecoderProfile::ffn_up_ms, stage_start);

        stage_start = ProfileClock::now();
        for (std::size_t index = 0; index < items.size(); ++index) {
            for (std::size_t dimension = 0; dimension < gate[index].size(); ++dimension) {
                gate[index][dimension] = gate[index][dimension] /
                    (1.0f + std::exp(-gate[index][dimension])) * up[index][dimension];
            }
        }
        add_profile_time(&RealLlamaDecoderProfile::ffn_activation_ms, stage_start);

        stage_start = ProfileClock::now();
        std::vector<std::vector<float>> ffn_projection;
        if (!linear_batch(weights.weights.ffn_down, gate, &ffn_projection, error,
                          policy_context->ffn_linear_policy, &batch_linear_profile)) {
            rollback();
            return false;
        }
        add_profile_time(&RealLlamaDecoderProfile::ffn_down_ms, stage_start);

        stage_start = ProfileClock::now();
        for (std::size_t index = 0; index < items.size(); ++index) {
            for (std::size_t dimension = 0; dimension < hidden[index].size(); ++dimension) {
                hidden[index][dimension] += ffn_projection[index][dimension];
            }
        }
        add_profile_time(&RealLlamaDecoderProfile::ffn_residual_ms, stage_start);
        add_profile_time(&RealLlamaDecoderProfile::ffn_ms, ffn_start);
    }

    auto stage_start = ProfileClock::now();
    std::vector<std::vector<float>> final_hidden(items.size());
    for (std::size_t index = 0; index < items.size(); ++index) {
        final_hidden[index] = rms_norm(hidden[index], impl_->data->output_norm,
                                       impl_->data->metadata.rms_norm_epsilon);
    }
    add_profile_time(&RealLlamaDecoderProfile::normalization_ms, stage_start);

    stage_start = ProfileClock::now();
    std::vector<std::vector<float>> result;
    if (!linear_batch(impl_->data->output_weight, final_hidden, &result, error,
                      policy_context->lm_head_linear_policy, &batch_linear_profile)) {
        rollback();
        return false;
    }
    add_profile_time(&RealLlamaDecoderProfile::lm_head_ms, stage_start);
    for (std::size_t index = 0; index < items.size(); ++index) {
        if (result[index].size() != impl_->data->metadata.vocab_size ||
            !std::all_of(result[index].begin(), result[index].end(),
                         [](float value) { return std::isfinite(value); })) {
            if (error != nullptr) *error = "batch Decoder produced invalid logits";
            rollback();
            return false;
        }
        if (snapshots[index].context->external_kv != nullptr &&
            !snapshots[index].context->external_kv->commit(
                snapshots[index].position + 1, error)) {
            rollback();
            return false;
        }
        ++snapshots[index].context->position;
        if (snapshots[index].context->profiling_enabled) {
            ++snapshots[index].context->profile.forward_tokens;
        }
    }
    const double batch_ms = std::chrono::duration<double, std::milli>(
        ProfileClock::now() - batch_start).count();
    for (const auto& snapshot : snapshots) {
        if (!snapshot.context->profiling_enabled) continue;
        snapshot.context->profile.total_ms += batch_ms;
        snapshot.context->profile.batch_linear_calls +=
            batch_linear_profile.batch_linear_calls;
        snapshot.context->profile.batch_input_pack_ms +=
            batch_linear_profile.batch_input_pack_ms;
        snapshot.context->profile.batch_kernel_ms += batch_linear_profile.batch_kernel_ms;
        snapshot.context->profile.batch_output_unpack_ms +=
            batch_linear_profile.batch_output_unpack_ms;
    }
    *logits = std::move(result);
    if (error != nullptr) error->clear();
    return true;
}

RealLlamaCpuContext::RealLlamaCpuContext(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}

RealLlamaCpuContext::~RealLlamaCpuContext() = default;
RealLlamaCpuContext::RealLlamaCpuContext(RealLlamaCpuContext&&) noexcept = default;
RealLlamaCpuContext& RealLlamaCpuContext::operator=(RealLlamaCpuContext&&) noexcept = default;

void RealLlamaCpuContext::reset() {
    if (impl_->external_kv != nullptr) impl_->external_kv->trim(0);
    impl_->position = 0;
    impl_->attention_score_count = 0;
    impl_->profile = {};
    for (auto& layer : impl_->layers) {
        layer.keys.clear();
        layer.values.clear();
    }
}

bool RealLlamaCpuContext::prefill(const TokenSequence& tokens,
                                  std::vector<float>* logits,
                                  std::string* error) {
    if (tokens.empty() || logits == nullptr) {
        if (error != nullptr) *error = "prefill requires tokens and logits output";
        return false;
    }
    reset();
    for (std::size_t i = 0; i < tokens.size(); ++i) {
        if (!impl_->forward_token(tokens[i], i + 1 == tokens.size() ? logits : nullptr,
                                  error)) return false;
    }
    return true;
}

bool RealLlamaCpuContext::decode(TokenId token,
                                 std::vector<float>* logits,
                                 std::string* error) {
    if (impl_->position == 0 || logits == nullptr) {
        if (error != nullptr) *error = "decode requires a completed prefill and logits output";
        return false;
    }
    return impl_->forward_token(token, logits, error);
}

bool RealLlamaCpuContext::forward_hidden_state(
    const std::vector<float>& input,
    std::vector<float>* output,
    std::string* error) {
    if (output == nullptr || input.size() != impl_->model->metadata.hidden_size) {
        if (error != nullptr) *error = "CPU hidden input or output is invalid";
        return false;
    }
    return impl_->forward_hidden(input, nullptr, output, error);
}

cpp_llm_engine::core::Status RealLlamaCpuContext::prefill_tokens(
    const cpp_llm_engine::core::DecoderTokenSequence& tokens,
    cpp_llm_engine::core::DecoderLogits* logits) {
    return prefill_chunk(tokens, true, true, logits);
}

cpp_llm_engine::core::Status RealLlamaCpuContext::prefill_chunk(
    const cpp_llm_engine::core::DecoderTokenSequence& tokens,
    bool start_new_sequence,
    bool produce_logits,
    cpp_llm_engine::core::DecoderLogits* logits) {
    if (tokens.empty() || (produce_logits && logits == nullptr)) {
        return cpp_llm_engine::core::Status::error(
            "prefill chunk requires tokens and final logits output");
    }
    if (start_new_sequence) {
        reset();
    } else if (impl_->position == 0) {
        return cpp_llm_engine::core::Status::error(
            "appended prefill chunk requires an existing sequence");
    }

    std::string error;
    for (std::size_t i = 0; i < tokens.size(); ++i) {
        const bool write_logits = produce_logits && i + 1 == tokens.size();
        if (!impl_->forward_token(tokens[i], write_logits ? logits : nullptr, &error)) {
            return cpp_llm_engine::core::Status::error(std::move(error));
        }
    }
    return cpp_llm_engine::core::Status::ok();
}

cpp_llm_engine::core::Status RealLlamaCpuContext::prefill_embeddings(
    const cpp_llm_engine::core::DecoderEmbeddingSequence& embeddings,
    cpp_llm_engine::core::DecoderLogits* logits) {
    const auto validation = embeddings.validate(impl_->model->metadata.hidden_size);
    if (!validation.is_ok()) return validation;
    if (logits == nullptr) {
        return cpp_llm_engine::core::Status::error(
            "embedding prefill requires final logits output");
    }
    reset();
    std::string error;
    for (std::size_t token = 0; token < embeddings.token_count; ++token) {
        const float* row = embeddings.values.data() + token * embeddings.hidden_size;
        const bool write_logits = token + 1 == embeddings.token_count;
        if (!impl_->forward_embedding(
                row, embeddings.hidden_size, write_logits ? logits : nullptr, &error)) {
            return cpp_llm_engine::core::Status::error(std::move(error));
        }
    }
    return cpp_llm_engine::core::Status::ok();
}

cpp_llm_engine::core::Status RealLlamaCpuContext::decode_token(
    cpp_llm_engine::core::DecoderTokenId token,
    cpp_llm_engine::core::DecoderLogits* logits) {
    std::string error;
    if (!decode(token, logits, &error)) {
        return cpp_llm_engine::core::Status::error(std::move(error));
    }
    return cpp_llm_engine::core::Status::ok();
}

std::size_t RealLlamaCpuContext::token_count() const {
    return impl_->position;
}

std::size_t RealLlamaCpuContext::kv_cache_bytes() const {
    if (impl_->external_kv != nullptr) {
        return impl_->position * impl_->model->layers.size() * impl_->model->kv_heads *
            impl_->model->head_dim * 2 * sizeof(float);
    }
    std::size_t values = 0;
    for (const auto& layer : impl_->layers) {
        values += layer.keys.size() + layer.values.size();
    }
    return values * sizeof(float);
}

bool RealLlamaCpuContext::copy_kv_token(
    std::size_t layer,
    std::size_t token_index,
    std::vector<float>* key,
    std::vector<float>* value,
    std::string* error) const {
    const auto fail = [&](const char* message) {
        if (key != nullptr) key->clear();
        if (value != nullptr) value->clear();
        if (error != nullptr) *error = message;
        return false;
    };
    if (key == nullptr || value == nullptr) return fail("KV copy outputs must not be null");
    if (layer >= impl_->layers.size() || token_index >= impl_->position) {
        return fail("KV copy index is outside the committed context");
    }
    const std::size_t width = impl_->model->kv_heads * impl_->model->head_dim;
    if (impl_->external_kv != nullptr) {
        key->resize(width);
        value->resize(width);
        for (std::size_t head = 0; head < impl_->model->kv_heads; ++head) {
            const float* source_key = impl_->external_kv->key_data(layer, token_index, head);
            const float* source_value = impl_->external_kv->value_data(layer, token_index, head);
            std::copy_n(source_key, impl_->model->head_dim,
                        key->begin() + head * impl_->model->head_dim);
            std::copy_n(source_value, impl_->model->head_dim,
                        value->begin() + head * impl_->model->head_dim);
        }
    } else {
        const auto& cache = impl_->layers[layer];
        const std::size_t offset = token_index * width;
        if (offset > cache.keys.size() || width > cache.keys.size() - offset ||
            offset > cache.values.size() || width > cache.values.size() - offset) {
            return fail("KV context payload does not match committed tokens");
        }
        key->assign(cache.keys.begin() + offset, cache.keys.begin() + offset + width);
        value->assign(
            cache.values.begin() + offset, cache.values.begin() + offset + width);
    }
    if (error != nullptr) error->clear();
    return true;
}

bool RealLlamaCpuContext::attach_kv_data_plane(
    std::shared_ptr<RealLlamaKvDataPlane> data_plane,
    std::string* error) {
    if (data_plane == nullptr) {
        if (error != nullptr) *error = "KV data plane must not be null";
        return false;
    }
    if (data_plane->token_count() != impl_->position) {
        if (error != nullptr) *error = "KV data plane token count does not match context";
        return false;
    }
    impl_->external_kv = std::move(data_plane);
    for (auto& layer : impl_->layers) {
        layer.keys.clear();
        layer.values.clear();
    }
    if (error != nullptr) error->clear();
    return true;
}

bool RealLlamaCpuContext::adopt_external_kv_prefix(
    std::size_t token_count,
    std::string* error) {
    if (impl_->external_kv == nullptr || impl_->position != 0 || token_count == 0 ||
        token_count >= impl_->model->metadata.max_position ||
        impl_->external_kv->token_count() != token_count) {
        if (error != nullptr) {
            *error = "external KV prefix does not match an empty model context";
        }
        return false;
    }
    impl_->position = token_count;
    if (error != nullptr) error->clear();
    return true;
}

bool RealLlamaCpuContext::uses_external_kv() const {
    return impl_->external_kv != nullptr;
}

cpp_llm_engine::core::DecoderModelContract RealLlamaCpuContext::model_contract() const {
    return {
        impl_->model->metadata.vocab_size,
        impl_->model->layers.size(),
        impl_->model->metadata.max_position,
        {
            impl_->model->metadata.hidden_size,
            impl_->model->layout.intermediate_size,
            impl_->model->query_heads,
            impl_->model->kv_heads,
            impl_->model->head_dim,
            impl_->model->rope_dim,
            cpp_llm_engine::core::DecoderNormalizationKind::RmsNorm,
            cpp_llm_engine::core::DecoderActivationKind::SwiGlu,
            cpp_llm_engine::core::DecoderPositionEncodingKind::Rotary,
            true,
            true,
        },
    };
}

void RealLlamaCpuContext::set_profiling_enabled(bool enabled) {
    impl_->profiling_enabled = enabled;
    impl_->profile = {};
}

bool RealLlamaCpuContext::set_attention_linear_policy(
    const chapter05_quantization::QuantizedLinearPolicy& policy,
    std::string* error) {
    const auto status = chapter05_quantization::validate_quantized_linear_policy(
        policy, available_quantized_linear_backends());
    if (!status.is_ok()) {
        if (error != nullptr) *error = status.message();
        return false;
    }
    impl_->attention_linear_policy = policy;
    if (error != nullptr) error->clear();
    return true;
}

bool RealLlamaCpuContext::set_ffn_linear_policy(
    const chapter05_quantization::QuantizedLinearPolicy& policy,
    std::string* error) {
    const auto status = chapter05_quantization::validate_quantized_linear_policy(
        policy, available_quantized_linear_backends());
    if (!status.is_ok()) {
        if (error != nullptr) *error = status.message();
        return false;
    }
    impl_->ffn_linear_policy = policy;
    if (error != nullptr) error->clear();
    return true;
}

RealLlamaDecoderStats RealLlamaCpuContext::stats() const {
    return RealLlamaDecoderStats{
        impl_->model->metadata.layer_count,
        impl_->model->layers.size(),
        impl_->position,
        impl_->model->query_heads,
        impl_->model->kv_heads,
        impl_->model->head_dim,
        impl_->model->rope_dim,
        impl_->attention_score_count,
        impl_->model->metadata.rope_freq_base,
    };
}

RealLlamaDecoderProfile RealLlamaCpuContext::profile() const {
    return impl_->profile;
}

}  // namespace chapter01
