// Model：实现 DeepSeek2 稠密层与 MoE 层的单词元 CPU 数值路径。
#include "cpp_llm_engine/model/real_deepseek2_dense_layer.h"

#include "cpp_llm_engine/core/linear_contract.h"
#include "chapter05_quantization/q2k_dequantize.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <set>

namespace cpp_llm_engine::model {
namespace {

bool fail(std::string* error, const std::string& message) {
    if (error != nullptr) *error = message;
    return false;
}

cpp_llm_engine::core::WeightFormat weight_format(chapter01::DType dtype) {
    using chapter01::DType;
    using cpp_llm_engine::core::WeightFormat;
    if (dtype == DType::Q2_K) return WeightFormat::Q2K;
    if (dtype == DType::Q3_K) return WeightFormat::Q3K;
    if (dtype == DType::IQ4_NL) return WeightFormat::IQ4_NL;
    return WeightFormat::Unknown;
}

bool linear(const chapter01::Tensor& weight,
            const std::vector<float>& input,
            const chapter05_quantization::QuantizedLinearPolicy& policy,
            std::vector<float>* output,
            std::string* error) {
    const auto& shape = weight.shape();
    if (shape.size() != 2 || shape[0] <= 0 || shape[1] <= 0 ||
        static_cast<std::size_t>(shape[0]) != input.size()) {
        return fail(error, "DeepSeek2 Linear shape does not match input");
    }
    const auto format = weight_format(weight.dtype());
    if (format == cpp_llm_engine::core::WeightFormat::Unknown) {
        return fail(error, "DeepSeek2 Linear weight format is not supported");
    }
    std::vector<float> values(static_cast<std::size_t>(shape[1]), 0.0F);
    const cpp_llm_engine::core::LinearRequest request{
        weight.raw_data(), weight.bytes(), input.data(), input.size(),
        values.data(), values.size(), values.size(), input.size(), format};
    const auto status =
        chapter05_quantization::compute_quantized_linear(request, policy);
    if (!status.is_ok()) return fail(error, status.message());
    *output = std::move(values);
    return true;
}

bool rms_norm(const std::vector<float>& input,
              const chapter01::Tensor& weight,
              float epsilon,
              std::vector<float>* output,
              std::string* error) {
    if (weight.dtype() != chapter01::DType::Float32 ||
        weight.numel() != input.size()) {
        return fail(error, "DeepSeek2 RMSNorm weight does not match input");
    }
    const auto scale = weight.to_vector<float>();
    double square_sum = 0.0;
    for (const float value : input) square_sum += value * value;
    const float inverse = 1.0F / std::sqrt(
        static_cast<float>(square_sum / input.size()) + epsilon);
    std::vector<float> values(input.size(), 0.0F);
    for (std::size_t i = 0; i < input.size(); ++i) {
        values[i] = input[i] * inverse * scale[i];
    }
    *output = std::move(values);
    return true;
}

float yarn_correction_dimension(std::size_t dimensions,
                                std::size_t original_context,
                                float rotations,
                                float frequency_base) {
    constexpr float kPi = 3.14159265358979323846F;
    return static_cast<float>(dimensions) * std::log(
        static_cast<float>(original_context) /
        (rotations * 2.0F * kPi)) / (2.0F * std::log(frequency_base));
}

void apply_rope(float* values,
                std::size_t dimensions,
                std::size_t position,
                const chapter01::ModelMetadata& metadata) {
    const bool yarn = metadata.rope_scaling_type == "yarn" &&
        metadata.rope_scaling_factor > 1.0F &&
        metadata.rope_scaling_original_context > 0;
    float correction_start = 0.0F;
    float correction_end = 0.0F;
    if (yarn) {
        correction_start = std::max(0.0F, std::floor(
            yarn_correction_dimension(
                dimensions, metadata.rope_scaling_original_context,
                32.0F, metadata.rope_freq_base)));
        correction_end = std::min(
            static_cast<float>(dimensions - 1), std::ceil(
                yarn_correction_dimension(
                    dimensions, metadata.rope_scaling_original_context,
                    1.0F, metadata.rope_freq_base)));
    }
    for (std::size_t pair = 0; pair < dimensions / 2; ++pair) {
        const float exponent =
            static_cast<float>(2 * pair) / static_cast<float>(dimensions);
        const float extrapolated = static_cast<float>(position) /
            std::pow(metadata.rope_freq_base, exponent);
        float angle = extrapolated;
        if (yarn) {
            const float interpolation =
                extrapolated / metadata.rope_scaling_factor;
            const float range = std::max(
                0.001F, correction_end - correction_start);
            const float normalized = std::clamp(
                (static_cast<float>(pair) - correction_start) / range,
                0.0F, 1.0F);
            const float ramp = 1.0F - normalized;
            angle = interpolation * (1.0F - ramp) + extrapolated * ramp;
        }
        const float cosine = std::cos(angle);
        const float sine = std::sin(angle);
        const float left = values[2 * pair];
        const float right = values[2 * pair + 1];
        values[2 * pair] = left * cosine - right * sine;
        values[2 * pair + 1] = left * sine + right * cosine;
    }
}

float dot(const float* left, const float* right, std::size_t count) {
    float value = 0.0F;
    for (std::size_t i = 0; i < count; ++i) value += left[i] * right[i];
    return value;
}

bool dequantize_kv_b(const chapter01::Tensor& weight,
                     std::vector<float>* values,
                     std::string* error) {
    if (weight.dtype() != chapter01::DType::Q2_K) {
        return fail(error, "DeepSeek2 combined K/V-B must use Q2_K");
    }
    return chapter05_quantization::dequantize_q2_k(
        reinterpret_cast<const std::uint8_t*>(weight.raw_data()),
        weight.bytes(), weight.numel(), values, error);
}

bool run_dense_ffn(
    const chapter01::RealDeepSeek2DenseLayerWeights& weights,
    const std::vector<float>& input,
    const chapter05_quantization::QuantizedLinearPolicy& policy,
    std::vector<float>* output,
    std::string* error) {
    std::vector<float> gate;
    std::vector<float> up;
    if (!linear(weights.ffn_gate, input, policy, &gate, error) ||
        !linear(weights.ffn_up, input, policy, &up, error) ||
        gate.size() != up.size()) {
        return fail(error, "DeepSeek2 dense FFN projections are inconsistent");
    }
    for (std::size_t i = 0; i < gate.size(); ++i) {
        gate[i] = gate[i] / (1.0F + std::exp(-gate[i])) * up[i];
    }
    return linear(weights.ffn_down, gate, policy, output, error);
}

template <typename Weights, typename FeedForward>
bool run_deepseek2_token_layer_impl(
    const chapter01::ModelMetadata& metadata,
    const Weights& weights,
    const std::vector<float>& input,
    RealDeepSeek2LayerContext* context,
    RealDeepSeek2LayerResult* result,
    std::string* error,
    const chapter05_quantization::QuantizedLinearPolicy& policy,
    FeedForward&& feed_forward) {
    if (result == nullptr) {
        return fail(error, "DeepSeek2 layer result must not be null");
    }
    if (context == nullptr) {
        return fail(error, "DeepSeek2 layer Context must not be null");
    }
    if (metadata.architecture != "deepseek2" ||
        input.size() != metadata.hidden_size ||
        metadata.attention_head_count == 0 ||
        metadata.attention_key_length_mla <= metadata.rope_dimension_count ||
        metadata.attention_value_length_mla == 0 ||
        metadata.attention_kv_lora_rank == 0) {
        return fail(error, "DeepSeek2 metadata or input is inconsistent");
    }
    if (!weights.valid() || !weights.uses_combined_kv_b()) {
        return fail(error,
                    "DeepSeek2 layer requires valid combined K/V-B weights");
    }

    const std::size_t latent = metadata.attention_kv_lora_rank;
    const std::size_t rope = metadata.rope_dimension_count;
    if (context->latent_kv.size() != context->position * latent ||
        context->key_rope.size() != context->position * rope) {
        return fail(error, "DeepSeek2 Context cache is not aligned with position");
    }

    RealDeepSeek2LayerResult built;
    if (!rms_norm(input, weights.attn_norm, metadata.rms_norm_epsilon,
                  &built.attention_norm, error) ||
        !linear(weights.attn_q, built.attention_norm, policy,
                &built.query, error) ||
        !linear(weights.attn_kv_a_mqa, built.attention_norm, policy,
                &built.compressed_kv_with_rope, error)) {
        return false;
    }
    if (built.compressed_kv_with_rope.size() != latent + rope) {
        return fail(error, "DeepSeek2 K/V-A output width is invalid");
    }
    std::vector<float> latent_values(
        built.compressed_kv_with_rope.begin(),
        built.compressed_kv_with_rope.begin() +
            static_cast<std::ptrdiff_t>(latent));
    if (!rms_norm(latent_values, weights.attn_kv_a_norm,
                  metadata.rms_norm_epsilon,
                  &built.normalized_latent_kv, error)) {
        return false;
    }

    const std::size_t heads = metadata.attention_head_count;
    const std::size_t nope = metadata.attention_key_length_mla - rope;
    const std::size_t value = metadata.attention_value_length_mla;
    if (built.query.size() != heads * (nope + rope)) {
        return fail(error, "DeepSeek2 Query width is invalid");
    }

    std::vector<float> query_nope(heads * nope, 0.0F);
    std::vector<float> query_rope(heads * rope, 0.0F);
    for (std::size_t head = 0; head < heads; ++head) {
        const float* query_head = built.query.data() + head * (nope + rope);
        std::copy(query_head, query_head + nope,
                  query_nope.begin() +
                      static_cast<std::ptrdiff_t>(head * nope));
        std::copy(query_head + nope, query_head + nope + rope,
                  query_rope.begin() +
                      static_cast<std::ptrdiff_t>(head * rope));
        apply_rope(query_rope.data() + head * rope, rope,
                   context->position, metadata);
    }
    std::vector<float> next_key_rope(
        built.compressed_kv_with_rope.begin() +
            static_cast<std::ptrdiff_t>(latent),
        built.compressed_kv_with_rope.end());
    apply_rope(next_key_rope.data(), rope, context->position, metadata);

    std::vector<float> kv_b;
    if (!dequantize_kv_b(weights.attn_kv_b, &kv_b, error)) return false;
    if (kv_b.size() != heads * (nope + value) * latent) {
        return fail(error, "DeepSeek2 K/V-B payload width is invalid");
    }
    std::vector<float> key_up(heads * nope * latent, 0.0F);
    std::vector<float> value_up(heads * value * latent, 0.0F);
    for (std::size_t head = 0; head < heads; ++head) {
        const std::size_t source_row = head * (nope + value);
        std::copy_n(kv_b.data() + source_row * latent, nope * latent,
                    key_up.data() + head * nope * latent);
        std::copy_n(kv_b.data() + (source_row + nope) * latent,
                    value * latent,
                    value_up.data() + head * value * latent);
    }

    auto candidate_latent = context->latent_kv;
    candidate_latent.insert(candidate_latent.end(),
                            built.normalized_latent_kv.begin(),
                            built.normalized_latent_kv.end());
    auto candidate_key_rope = context->key_rope;
    candidate_key_rope.insert(candidate_key_rope.end(),
                              next_key_rope.begin(), next_key_rope.end());
    const std::size_t key_tokens = context->position + 1;
    built.attention_value.assign(heads * value, 0.0F);
    float scale = 1.0F / std::sqrt(static_cast<float>(nope + rope));
    if (metadata.rope_scaling_type == "yarn" &&
        metadata.rope_scaling_factor > 1.0F &&
        metadata.rope_yarn_log_multiplier > 0.0F) {
        const float restored_log_multiplier =
            metadata.rope_yarn_log_multiplier / 0.1F;
        const float mscale = 1.0F + 0.1F * restored_log_multiplier *
            std::log(metadata.rope_scaling_factor);
        scale *= mscale * mscale;
    }
    std::vector<float> scores(key_tokens, 0.0F);
    for (std::size_t head = 0; head < heads; ++head) {
        float maximum = -std::numeric_limits<float>::infinity();
        for (std::size_t token = 0; token < key_tokens; ++token) {
            const float* latent_row = candidate_latent.data() + token * latent;
            float score = dot(query_rope.data() + head * rope,
                              candidate_key_rope.data() + token * rope, rope);
            for (std::size_t dim = 0; dim < nope; ++dim) {
                score += query_nope[head * nope + dim] *
                    dot(key_up.data() + (head * nope + dim) * latent,
                        latent_row, latent);
            }
            scores[token] = score * scale;
            maximum = std::max(maximum, scores[token]);
        }
        float denominator = 0.0F;
        for (std::size_t token = 0; token < key_tokens; ++token) {
            scores[token] = std::exp(scores[token] - maximum);
            denominator += scores[token];
        }
        if (!std::isfinite(denominator) || denominator <= 0.0F) {
            return fail(error,
                        "DeepSeek2 MLA Softmax denominator is invalid");
        }
        for (std::size_t token = 0; token < key_tokens; ++token) {
            const float probability = scores[token] / denominator;
            const float* latent_row = candidate_latent.data() + token * latent;
            for (std::size_t dim = 0; dim < value; ++dim) {
                built.attention_value[head * value + dim] += probability *
                    dot(value_up.data() + (head * value + dim) * latent,
                        latent_row, latent);
            }
        }
    }
    if (!linear(weights.attn_output, built.attention_value, policy,
                &built.attention_output, error)) {
        return false;
    }
    built.attention_residual.resize(input.size());
    for (std::size_t i = 0; i < input.size(); ++i) {
        built.attention_residual[i] = input[i] + built.attention_output[i];
    }
    if (!rms_norm(built.attention_residual, weights.ffn_norm,
                  metadata.rms_norm_epsilon, &built.ffn_norm, error) ||
        !feed_forward(built.ffn_norm, &built.ffn_output, error)) {
        return false;
    }
    if (built.ffn_output.size() != input.size()) {
        return fail(error, "DeepSeek2 FFN output width is invalid");
    }
    built.output.resize(input.size());
    for (std::size_t i = 0; i < input.size(); ++i) {
        built.output[i] = built.attention_residual[i] + built.ffn_output[i];
    }

    context->position = key_tokens;
    context->latent_kv = std::move(candidate_latent);
    context->key_rope = std::move(candidate_key_rope);
    *result = std::move(built);
    if (error != nullptr) error->clear();
    return true;
}

}  // namespace

void RealDeepSeek2LayerContext::reset() {
    position = 0;
    latent_kv.clear();
    key_rope.clear();
}

bool run_deepseek2_dense_token_layer(
    const chapter01::ModelMetadata& metadata,
    const chapter01::RealDeepSeek2DenseLayerWeights& weights,
    const std::vector<float>& input,
    RealDeepSeek2LayerContext* context,
    RealDeepSeek2LayerResult* result,
    std::string* error,
    const chapter05_quantization::QuantizedLinearPolicy& policy) {
    return run_deepseek2_token_layer_impl(
        metadata, weights, input, context, result, error, policy,
        [&weights, &policy](const std::vector<float>& normalized,
                            std::vector<float>* output,
                            std::string* local_error) {
            return run_dense_ffn(
                weights, normalized, policy, output, local_error);
        });
}

bool run_deepseek2_dense_first_token_layer(
    const chapter01::ModelMetadata& metadata,
    const chapter01::RealDeepSeek2DenseLayerWeights& weights,
    const std::vector<float>& input,
    RealDeepSeek2LayerResult* result,
    std::string* error,
    const chapter05_quantization::QuantizedLinearPolicy& policy) {
    RealDeepSeek2LayerContext context;
    return run_deepseek2_dense_token_layer(
        metadata, weights, input, &context, result, error, policy);
}

bool run_deepseek2_moe_token_layer(
    const chapter01::ModelMetadata& metadata,
    const chapter01::RealDeepSeek2MoeLayerWeights& weights,
    const std::vector<float>& input,
    RealDeepSeek2LayerContext* context,
    RealDeepSeek2LayerResult* result,
    RealMoeRoute* route,
    std::string* error,
    const chapter05_quantization::QuantizedLinearPolicy& policy,
    RealMoeTrace* trace) {
    if (route == nullptr) {
        return fail(error, "DeepSeek2 MoE route must not be null");
    }
    RealMoeRoute built_route;
    RealMoeTrace built_trace;
    const bool success = run_deepseek2_token_layer_impl(
        metadata, weights, input, context, result, error, policy,
        [&metadata, &weights, &policy, &built_route, &built_trace](
            const std::vector<float>& normalized,
            std::vector<float>* output,
            std::string* local_error) {
            return run_real_deepseek2_moe_feed_forward(
                metadata, weights, normalized, output, &built_route,
                local_error, policy, &built_trace);
        });
    if (!success) return false;
    *route = std::move(built_route);
    if (trace != nullptr) *trace = std::move(built_trace);
    return true;
}

bool run_deepseek2_moe_token_layer_batch(
    const chapter01::ModelMetadata& metadata,
    const chapter01::RealDeepSeek2MoeLayerWeights& weights,
    const std::vector<std::vector<float>>& inputs,
    const std::vector<RealDeepSeek2LayerContext*>& contexts,
    std::vector<RealDeepSeek2LayerResult>* results,
    std::vector<RealMoeRoute>* routes,
    RealMoeDispatchStats* stats,
    std::string* error,
    const chapter05_quantization::QuantizedLinearPolicy& policy,
    RealMoeBatchMode mode) {
    if (results == nullptr || routes == nullptr || stats == nullptr ||
        inputs.empty() || inputs.size() != contexts.size()) {
        return fail(error, "DeepSeek2 MoE layer batch arguments are invalid");
    }

    std::set<RealDeepSeek2LayerContext*> unique_contexts;
    std::vector<RealDeepSeek2LayerContext> saved_contexts;
    saved_contexts.reserve(contexts.size());
    for (auto* context : contexts) {
        if (context == nullptr || !unique_contexts.insert(context).second) {
            return fail(error, "DeepSeek2 MoE layer batch Context is invalid");
        }
        saved_contexts.push_back(*context);
    }
    const auto rollback = [&]() {
        for (std::size_t index = 0; index < contexts.size(); ++index) {
            *contexts[index] = saved_contexts[index];
        }
    };

    std::vector<RealDeepSeek2LayerResult> built_results(inputs.size());
    std::vector<std::vector<float>> normalized(inputs.size());
    for (std::size_t request = 0; request < inputs.size(); ++request) {
        const bool success = run_deepseek2_token_layer_impl(
            metadata, weights, inputs[request], contexts[request],
            &built_results[request], error, policy,
            [&normalized, request](const std::vector<float>& ffn_input,
                                   std::vector<float>* output,
                                   std::string*) {
                normalized[request] = ffn_input;
                output->assign(ffn_input.size(), 0.0F);
                return true;
            });
        if (!success) {
            rollback();
            return false;
        }
    }

    std::vector<std::vector<float>> ffn_outputs;
    std::vector<RealMoeRoute> built_routes;
    RealMoeDispatchStats built_stats;
    if (!run_real_deepseek2_moe_feed_forward_batch(
            metadata, weights, normalized, &ffn_outputs, &built_routes,
            &built_stats, error, policy, mode)) {
        rollback();
        return false;
    }
    if (ffn_outputs.size() != built_results.size()) {
        rollback();
        return fail(error, "DeepSeek2 MoE layer batch output count is invalid");
    }
    for (std::size_t request = 0; request < built_results.size(); ++request) {
        auto& built = built_results[request];
        if (ffn_outputs[request].size() != built.attention_residual.size()) {
            rollback();
            return fail(error,
                        "DeepSeek2 MoE layer batch output width is invalid");
        }
        built.ffn_output = std::move(ffn_outputs[request]);
        built.output.resize(built.attention_residual.size());
        for (std::size_t column = 0; column < built.output.size(); ++column) {
            built.output[column] =
                built.attention_residual[column] + built.ffn_output[column];
        }
    }

    *results = std::move(built_results);
    *routes = std::move(built_routes);
    *stats = built_stats;
    if (error != nullptr) error->clear();
    return true;
}

}  // namespace cpp_llm_engine::model
