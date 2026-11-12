// DeepSeek2 CUDA Model：实现稠密层的设备端单词元前向与缓存回退。
#include "cpp_llm_engine/model/cuda_deepseek2_dense_layer.h"

#include "cpp_llm_engine/cuda/cuda_execution_context.h"
#include "cpp_llm_engine/model/cuda_deepseek2_dense_layer_weights.h"

#include <cmath>
#include <utility>

namespace cpp_llm_engine::model {
namespace {

core::Status make_tensor(
    const std::vector<std::size_t>& shape, cuda::CudaTensor* output) {
    return cuda::CudaTensor::create_contiguous(
        shape, core::DType::Float32, output);
}

}  // namespace

core::Status CudaDeepSeek2DenseLayerConfig::validate() const {
    if (hidden_size == 0 || head_count == 0 || nope_dimension == 0 ||
        rope_dimension == 0 || rope_dimension % 2 != 0 ||
        value_dimension == 0 || latent_dimension == 0 || ffn_dimension == 0 ||
        !std::isfinite(rms_norm_epsilon) || rms_norm_epsilon <= 0.0F ||
        !std::isfinite(rope_frequency_base) || rope_frequency_base <= 0.0F ||
        !std::isfinite(attention_scale) || attention_scale <= 0.0F) {
        return core::Status::error("DeepSeek2 CUDA layer config is invalid");
    }
    return yarn.validate();
}

core::Status CudaDeepSeek2DenseLayer::create(
    const CudaDeepSeek2DenseLayerConfig& config,
    std::size_t capacity_tokens,
    CudaDeepSeek2DenseLayer* output) {
    if (output == nullptr || !config.validate().is_ok()) {
        return core::Status::error("DeepSeek2 CUDA layer output/config is invalid");
    }
    CudaDeepSeek2DenseLayer replacement;
    replacement.config_ = config;
    core::Status status = cuda::CudaMlaCache::create(
        capacity_tokens, config.latent_dimension, config.rope_dimension,
        &replacement.cache_);
    const auto create = [&status](
        cuda::CudaTensor* tensor, std::vector<std::size_t> shape) {
        if (status.is_ok()) status = make_tensor(shape, tensor);
    };
    create(&replacement.attention_norm_, {config.hidden_size});
    create(&replacement.query_, {config.head_count,
                                  config.nope_dimension + config.rope_dimension});
    create(&replacement.compressed_kv_,
           {config.latent_dimension + config.rope_dimension});
    create(&replacement.query_nope_, {config.head_count, config.nope_dimension});
    create(&replacement.query_rope_,
           {1, config.head_count, config.rope_dimension});
    create(&replacement.latent_, {config.latent_dimension});
    create(&replacement.normalized_latent_, {config.latent_dimension});
    create(&replacement.key_rope_, {1, 1, config.rope_dimension});
    create(&replacement.attention_probabilities_,
           {config.head_count, capacity_tokens});
    create(&replacement.attention_value_,
           {config.head_count, config.value_dimension});
    create(&replacement.attention_output_, {config.hidden_size});
    create(&replacement.attention_residual_, {config.hidden_size});
    create(&replacement.ffn_norm_, {config.hidden_size});
    create(&replacement.gate_, {config.ffn_dimension});
    create(&replacement.up_, {config.ffn_dimension});
    create(&replacement.activated_, {config.ffn_dimension});
    create(&replacement.ffn_output_, {config.hidden_size});
    if (!status.is_ok()) return status;
    *output = std::move(replacement);
    return core::Status::ok();
}

core::Status CudaDeepSeek2DenseLayer::forward(
    cuda::CudaExecutionContext* context,
    const CudaDeepSeek2DenseLayerWeightView& weights,
    cuda::CudaTensor* hidden,
    std::size_t position) {
    if (context == nullptr || hidden == nullptr || !weights.valid() ||
        position != cache_.size_tokens() ||
        hidden->contract().shape.dimensions !=
            std::vector<std::size_t>{config_.hidden_size}) {
        return core::Status::error("DeepSeek2 CUDA layer input/context is invalid");
    }
    core::Status status = context->rms_norm(
        *hidden, *weights.attention_norm, &attention_norm_,
        config_.rms_norm_epsilon);
    if (status.is_ok()) status = context->linear(
        attention_norm_, weights.query, &query_);
    if (status.is_ok()) status = context->linear(
        attention_norm_, weights.kv_a_mqa, &compressed_kv_);
    if (status.is_ok()) status = context->prepare_mla(
        query_, compressed_kv_, &query_nope_, &query_rope_, &latent_,
        &key_rope_);
    if (status.is_ok()) status = context->rms_norm(
        latent_, *weights.kv_a_norm, &normalized_latent_,
        config_.rms_norm_epsilon);
    if (status.is_ok()) status = context->rope(
        &query_rope_, position, config_.rope_dimension,
        config_.rope_frequency_base, core::RotaryPairing::Adjacent,
        config_.yarn);
    if (status.is_ok()) status = context->rope(
        &key_rope_, position, config_.rope_dimension,
        config_.rope_frequency_base, core::RotaryPairing::Adjacent,
        config_.yarn);
    if (status.is_ok()) status = context->append_mla(
        &cache_, normalized_latent_, key_rope_);
    if (status.is_ok()) status = context->mla_attention(
        query_nope_, query_rope_, *weights.kv_b_dequantized, cache_,
        &attention_probabilities_, &attention_value_, config_.attention_scale);
    if (status.is_ok()) status = context->linear(
        attention_value_, weights.attention_output, &attention_output_);
    if (status.is_ok()) status = context->vector_add(
        *hidden, attention_output_, &attention_residual_);
    if (status.is_ok()) status = context->rms_norm(
        attention_residual_, *weights.ffn_norm, &ffn_norm_,
        config_.rms_norm_epsilon);
    if (status.is_ok()) status = context->linear(
        ffn_norm_, weights.ffn_gate, &gate_);
    if (status.is_ok()) status = context->linear(
        ffn_norm_, weights.ffn_up, &up_);
    if (status.is_ok()) status = context->swiglu(gate_, up_, &activated_);
    if (status.is_ok()) status = context->linear(
        activated_, weights.ffn_down, &ffn_output_);
    if (status.is_ok()) status = context->vector_add(
        attention_residual_, ffn_output_, hidden);
    if (!status.is_ok()) cache_.trim(position);
    return status;
}

void CudaDeepSeek2DenseLayer::reset() { cache_.clear(); }
void CudaDeepSeek2DenseLayer::trim(std::size_t position) {
    cache_.trim(position);
}
std::size_t CudaDeepSeek2DenseLayer::position() const {
    return cache_.size_tokens();
}
const cuda::CudaMlaCache& CudaDeepSeek2DenseLayer::cache() const {
    return cache_;
}

}  // namespace cpp_llm_engine::model
