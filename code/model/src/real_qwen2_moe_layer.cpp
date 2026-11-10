// Model：实现 Qwen2MoE 单层多词元 Decoder CPU 执行。
// 职责：执行 RMSNorm、NeoX RoPE、因果 Attention、MoE 与残差。
// 契约：K/V 按词元连续返回，查询头按 GQA 分组映射到 K/V 头。
// 验证：qwen15_moe_ffn_compare 的逐层 hidden、Top-K 与 K/V 对照。
#include "cpp_llm_engine/model/real_qwen2_moe_layer.h"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace cpp_llm_engine::model {
namespace {

bool fail(std::string* error, const std::string& message) {
    if (error != nullptr) error->assign(message);
    return false;
}

bool float_values(const chapter01::Tensor& tensor,
                  std::vector<float>* output,
                  std::string* error) {
    if (tensor.dtype() != chapter01::DType::Float32 ||
        tensor.raw_data() == nullptr) {
        return fail(error, "Qwen2MoE norm and bias tensors must be Float32");
    }
    std::vector<float> values(tensor.numel(), 0.0F);
    std::memcpy(values.data(), tensor.raw_data(), values.size() * sizeof(float));
    *output = std::move(values);
    return true;
}

cpp_llm_engine::core::WeightFormat weight_format(chapter01::DType dtype) {
    using chapter01::DType;
    using cpp_llm_engine::core::WeightFormat;
    if (dtype == DType::Q3_K) return WeightFormat::Q3K;
    if (dtype == DType::Q4_K) return WeightFormat::Q4K;
    if (dtype == DType::Q5_0) return WeightFormat::Q5_0;
    if (dtype == DType::Q5_1) return WeightFormat::Q5_1;
    if (dtype == DType::Q5_K) return WeightFormat::Q5K;
    if (dtype == DType::Q6_K) return WeightFormat::Q6K;
    if (dtype == DType::Q8_0) return WeightFormat::Q8_0;
    return WeightFormat::Unknown;
}

bool linear(const chapter01::Tensor& weight,
            const std::vector<float>& input,
            const chapter05_quantization::QuantizedLinearPolicy& policy,
            std::vector<float>* output,
            std::string* error) {
    const auto& shape = weight.shape();
    if (shape.size() != 2 || shape[0] <= 0 || shape[1] <= 0) {
        return fail(error, "Qwen2MoE Linear weight must be a positive matrix");
    }
    const auto cols = static_cast<std::size_t>(shape[0]);
    const auto rows = static_cast<std::size_t>(shape[1]);
    const auto format = weight_format(weight.dtype());
    if (format == cpp_llm_engine::core::WeightFormat::Unknown) {
        return fail(error, "Qwen2MoE Linear weight format is not supported");
    }
    std::vector<float> values(rows, 0.0F);
    const cpp_llm_engine::core::LinearRequest request{
        weight.raw_data(), weight.bytes(), input.data(), input.size(),
        values.data(), values.size(), rows, cols, format};
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
    std::vector<float> scale;
    if (!float_values(weight, &scale, error) || scale.size() != input.size()) {
        return fail(error, "Qwen2MoE RMSNorm weight does not match hidden size");
    }
    double square_sum = 0.0;
    for (const float value : input) square_sum += value * value;
    const float inverse = 1.0F / std::sqrt(
        static_cast<float>(square_sum / input.size()) + epsilon);
    std::vector<float> normalized(input.size(), 0.0F);
    for (std::size_t i = 0; i < input.size(); ++i) {
        normalized[i] = input[i] * inverse * scale[i];
    }
    *output = std::move(normalized);
    return true;
}

bool project_with_bias(
    const chapter01::Tensor& weight,
    const chapter01::Tensor& bias,
    const std::vector<float>& input,
    const chapter05_quantization::QuantizedLinearPolicy& policy,
    std::vector<float>* output,
    std::string* error) {
    std::vector<float> projected;
    std::vector<float> bias_values;
    if (!linear(weight, input, policy, &projected, error) ||
        !float_values(bias, &bias_values, error) ||
        projected.size() != bias_values.size()) {
        return fail(error, "Qwen2MoE Attention projection bias does not match output");
    }
    for (std::size_t i = 0; i < projected.size(); ++i) {
        projected[i] += bias_values[i];
    }
    *output = std::move(projected);
    return true;
}

void apply_neox_rope(std::vector<float>* values,
                     std::size_t head_count,
                     std::size_t head_dim,
                     std::size_t position,
                     float frequency_base) {
    for (std::size_t head = 0; head < head_count; ++head) {
        float* vector = values->data() + head * head_dim;
        for (std::size_t pair = 0; pair < head_dim / 2; ++pair) {
            const float exponent = -2.0F * static_cast<float>(pair) /
                static_cast<float>(head_dim);
            const float angle = static_cast<float>(position) *
                std::pow(frequency_base, exponent);
            const float cosine = std::cos(angle);
            const float sine = std::sin(angle);
            const std::size_t second = pair + head_dim / 2;
            const float first_value = vector[pair];
            const float second_value = vector[second];
            vector[pair] = first_value * cosine - second_value * sine;
            vector[second] = first_value * sine + second_value * cosine;
        }
    }
}

float dot(const float* left, const float* right, std::size_t size) {
    float value = 0.0F;
    for (std::size_t i = 0; i < size; ++i) value += left[i] * right[i];
    return value;
}

void softmax(std::vector<float>* values) {
    const float maximum = *std::max_element(values->begin(), values->end());
    float denominator = 0.0F;
    for (float& value : *values) {
        value = std::exp(value - maximum);
        denominator += value;
    }
    for (float& value : *values) value /= denominator;
}

struct PendingLayer {
    RealQwen2MoePrefillResult result;
    std::vector<std::vector<float>> ffn_inputs;
};

bool prepare_attention(
    const chapter01::ModelMetadata& metadata,
    const chapter01::RealMoeLayerWeights& weights,
    const std::vector<std::vector<float>>& inputs,
    const RealQwen2MoeLayerContext& context,
    PendingLayer* pending,
    std::string* error,
    const chapter05_quantization::QuantizedLinearPolicy& policy) {
    if (pending == nullptr || metadata.architecture != "qwen2moe" ||
        inputs.empty() || metadata.attention_head_count == 0 ||
        metadata.attention_head_count_kv == 0 ||
        metadata.attention_head_count % metadata.attention_head_count_kv != 0 ||
        metadata.hidden_size % metadata.attention_head_count != 0) {
        return fail(error, "Qwen2MoE prefill metadata is inconsistent");
    }
    for (const auto& input : inputs) {
        if (input.size() != metadata.hidden_size) {
            return fail(error, "Qwen2MoE prefill input width is inconsistent");
        }
    }
    const std::size_t head_count = metadata.attention_head_count;
    const std::size_t kv_head_count = metadata.attention_head_count_kv;
    const std::size_t token_count = inputs.size();
    std::vector<std::vector<float>> queries(token_count);
    std::vector<std::vector<float>> keys(token_count);
    std::vector<std::vector<float>> values(token_count);
    for (std::size_t token = 0; token < token_count; ++token) {
        std::vector<float> normalized;
        if (!rms_norm(inputs[token], weights.attn_norm,
                      metadata.rms_norm_epsilon, &normalized, error) ||
            !project_with_bias(weights.attn_q, weights.attn_q_bias,
                               normalized, policy, &queries[token], error) ||
            !project_with_bias(weights.attn_k, weights.attn_k_bias,
                               normalized, policy, &keys[token], error) ||
            !project_with_bias(weights.attn_v, weights.attn_v_bias,
                               normalized, policy, &values[token], error)) {
            return false;
        }
        if (queries[token].size() % head_count != 0 ||
            keys[token].size() % kv_head_count != 0 ||
            values[token].size() % kv_head_count != 0 ||
            queries[token].size() / head_count !=
                keys[token].size() / kv_head_count) {
            return fail(error, "Qwen2MoE Attention head widths are inconsistent");
        }
        const std::size_t query_head_dim = queries[token].size() / head_count;
        const std::size_t key_head_dim = keys[token].size() / kv_head_count;
        apply_neox_rope(&queries[token], head_count, query_head_dim,
                        context.position + token, metadata.rope_freq_base);
        apply_neox_rope(&keys[token], kv_head_count, key_head_dim,
                        context.position + token, metadata.rope_freq_base);
    }

    const std::size_t key_width = keys.front().size();
    const std::size_t value_width = values.front().size();
    if (context.keys.size() != context.position * key_width ||
        context.values.size() != context.position * value_width) {
        return fail(error, "Qwen2MoE layer Context K/V shape is inconsistent");
    }

    PendingLayer built;
    built.result.keys.reserve(token_count * key_width);
    built.result.values.reserve(token_count * value_width);
    built.result.tokens.resize(token_count);
    built.ffn_inputs.resize(token_count);
    for (std::size_t token = 0; token < token_count; ++token) {
        built.result.keys.insert(
            built.result.keys.end(), keys[token].begin(), keys[token].end());
        built.result.values.insert(
            built.result.values.end(), values[token].begin(), values[token].end());
        const std::size_t query_head_dim = queries[token].size() / head_count;
        const std::size_t value_head_dim = values[token].size() / kv_head_count;
        const std::size_t query_heads_per_kv_head = head_count / kv_head_count;
        std::vector<float> attention_context(
            head_count * value_head_dim, 0.0F);
        for (std::size_t head = 0; head < head_count; ++head) {
            const std::size_t kv_head = head / query_heads_per_kv_head;
            const std::size_t visible_tokens = context.position + token + 1;
            std::vector<float> scores(visible_tokens, 0.0F);
            for (std::size_t source = 0; source < visible_tokens; ++source) {
                const float* key = source < context.position
                    ? context.keys.data() + source * key_width
                    : keys[source - context.position].data();
                scores[source] = dot(
                    queries[token].data() + head * query_head_dim,
                    key + kv_head * query_head_dim, query_head_dim) /
                    std::sqrt(static_cast<float>(query_head_dim));
            }
            softmax(&scores);
            for (std::size_t source = 0; source < visible_tokens; ++source) {
                const float* value = source < context.position
                    ? context.values.data() + source * value_width
                    : values[source - context.position].data();
                for (std::size_t column = 0; column < value_head_dim; ++column) {
                    attention_context[head * value_head_dim + column] +=
                        scores[source] * value[kv_head * value_head_dim + column];
                }
            }
        }
        std::vector<float> attention_output;
        if (!linear(weights.attn_output, attention_context, policy,
                    &attention_output, error)) {
            return false;
        }
        auto& token_result = built.result.tokens[token];
        token_result.attention_residual.resize(metadata.hidden_size);
        for (std::size_t column = 0; column < metadata.hidden_size; ++column) {
            token_result.attention_residual[column] =
                inputs[token][column] + attention_output[column];
        }
        if (!rms_norm(token_result.attention_residual, weights.ffn_norm,
                      metadata.rms_norm_epsilon, &built.ffn_inputs[token],
                      error)) {
            return false;
        }
    }
    *pending = std::move(built);
    return true;
}

bool finish_moe(
    const chapter01::ModelMetadata& metadata,
    PendingLayer* pending,
    std::vector<std::vector<float>> moe_outputs,
    std::vector<RealMoeRoute> routes,
    std::string* error) {
    if (pending == nullptr ||
        pending->result.tokens.size() != moe_outputs.size() ||
        routes.size() != moe_outputs.size()) {
        return fail(error, "Qwen2MoE MoE result count is inconsistent");
    }
    for (std::size_t token = 0; token < moe_outputs.size(); ++token) {
        auto& result = pending->result.tokens[token];
        result.moe_output = std::move(moe_outputs[token]);
        result.route = std::move(routes[token]);
        result.output.resize(metadata.hidden_size);
        for (std::size_t column = 0; column < metadata.hidden_size; ++column) {
            result.output[column] = result.attention_residual[column] +
                result.moe_output[column];
            if (!std::isfinite(result.output[column])) {
                return fail(error, "Qwen2MoE prefill produced a non-finite value");
            }
        }
    }
    return true;
}

void commit_pending(const PendingLayer& pending,
                    RealQwen2MoeLayerContext* context) {
    context->keys.insert(context->keys.end(), pending.result.keys.begin(),
                         pending.result.keys.end());
    context->values.insert(context->values.end(), pending.result.values.begin(),
                           pending.result.values.end());
    context->position += pending.result.tokens.size();
}

}  // namespace

bool run_qwen2_moe_first_token_layer(
    const chapter01::ModelMetadata& metadata,
    const chapter01::RealMoeLayerWeights& weights,
    const std::vector<float>& input,
    RealQwen2MoeLayerResult* result,
    std::string* error,
    const chapter05_quantization::QuantizedLinearPolicy& policy) {
    RealQwen2MoePrefillResult prefill;
    if (!run_qwen2_moe_prefill_layer(
            metadata, weights, {input}, &prefill, error, policy)) {
        return false;
    }
    *result = std::move(prefill.tokens.front());
    return true;
}

bool run_qwen2_moe_prefill_layer(
    const chapter01::ModelMetadata& metadata,
    const chapter01::RealMoeLayerWeights& weights,
    const std::vector<std::vector<float>>& inputs,
    RealQwen2MoePrefillResult* result,
    std::string* error,
    const chapter05_quantization::QuantizedLinearPolicy& policy) {
    RealQwen2MoeLayerContext context;
    return run_qwen2_moe_layer_with_context(
        metadata, weights, inputs, &context, result, error, policy);
}

void RealQwen2MoeLayerContext::reset() {
    position = 0;
    keys.clear();
    values.clear();
}

bool run_qwen2_moe_layer_with_context(
    const chapter01::ModelMetadata& metadata,
    const chapter01::RealMoeLayerWeights& weights,
    const std::vector<std::vector<float>>& inputs,
    RealQwen2MoeLayerContext* context,
    RealQwen2MoePrefillResult* result,
    std::string* error,
    const chapter05_quantization::QuantizedLinearPolicy& policy) {
    if (context == nullptr || result == nullptr) {
        return fail(error, "Qwen2MoE prefill output or Context is missing");
    }
    PendingLayer pending;
    if (!prepare_attention(
            metadata, weights, inputs, *context, &pending, error, policy)) {
        return false;
    }
    std::vector<std::vector<float>> moe_outputs;
    std::vector<RealMoeRoute> routes;
    if (!run_real_moe_feed_forward_batch(
            metadata, weights, pending.ffn_inputs, &moe_outputs, &routes, nullptr,
            error, policy)) {
        return false;
    }
    if (!finish_moe(metadata, &pending, std::move(moe_outputs),
                    std::move(routes), error)) {
        return false;
    }
    commit_pending(pending, context);
    *result = std::move(pending.result);
    if (error != nullptr) error->clear();
    return true;
}

bool run_qwen2_moe_layer_batch_with_context(
    const chapter01::ModelMetadata& metadata,
    const chapter01::RealMoeLayerWeights& weights,
    const std::vector<std::vector<float>>& inputs,
    const std::vector<RealQwen2MoeLayerContext*>& contexts,
    std::vector<RealQwen2MoeLayerResult>* results,
    RealMoeDispatchStats* dispatch_stats,
    std::string* error,
    const chapter05_quantization::QuantizedLinearPolicy& policy,
    RealMoeBatchMode batch_mode) {
    if (results == nullptr || inputs.empty() || inputs.size() != contexts.size()) {
        return fail(error, "Qwen2MoE layer batch sizes are inconsistent");
    }
    std::vector<PendingLayer> pending(inputs.size());
    std::vector<std::vector<float>> ffn_inputs;
    ffn_inputs.reserve(inputs.size());
    for (std::size_t index = 0; index < inputs.size(); ++index) {
        if (contexts[index] == nullptr ||
            !prepare_attention(metadata, weights, {inputs[index]},
                               *contexts[index], &pending[index], error,
                               policy)) {
            return false;
        }
        ffn_inputs.push_back(pending[index].ffn_inputs.front());
    }
    std::vector<std::vector<float>> moe_outputs;
    std::vector<RealMoeRoute> routes;
    RealMoeDispatchStats built_stats;
    if (!run_real_moe_feed_forward_batch(
            metadata, weights, ffn_inputs, &moe_outputs, &routes,
            &built_stats, error, policy, batch_mode)) {
        return false;
    }
    std::vector<RealQwen2MoeLayerResult> built;
    built.reserve(inputs.size());
    for (std::size_t index = 0; index < pending.size(); ++index) {
        std::vector<std::vector<float>> output{std::move(moe_outputs[index])};
        std::vector<RealMoeRoute> route{std::move(routes[index])};
        if (!finish_moe(metadata, &pending[index], std::move(output),
                        std::move(route), error)) {
            return false;
        }
        built.push_back(std::move(pending[index].result.tokens.front()));
    }
    for (std::size_t index = 0; index < pending.size(); ++index) {
        commit_pending(pending[index], contexts[index]);
    }
    *results = std::move(built);
    if (dispatch_stats != nullptr) *dispatch_stats = built_stats;
    if (error != nullptr) error->clear();
    return true;
}

}  // namespace cpp_llm_engine::model
