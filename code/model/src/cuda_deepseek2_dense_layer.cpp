// DeepSeek2 CUDA Model：实现稠密层的设备端单词元前向与缓存回退。
#include "cpp_llm_engine/model/cuda_deepseek2_dense_layer.h"

#include "cpp_llm_engine/cuda/cuda_execution_context.h"
#include "cpp_llm_engine/model/cuda_deepseek2_dense_layer_weights.h"

#include <cmath>
#include <limits>
#include <utility>

namespace cpp_llm_engine::model {
namespace {

core::Status make_tensor(
    const std::vector<std::size_t>& shape, cuda::CudaTensor* output) {
    return cuda::CudaTensor::create_contiguous(
        shape, core::DType::Float32, output);
}

}  // namespace

bool CudaDeepSeek2PagedMlaView::valid() const {
    return cache != nullptr && cache->valid() && physical_blocks != nullptr &&
        !physical_blocks->empty() && layer < cache->layer_count() &&
        physical_block < cache->total_blocks() && slot < cache->block_size() &&
        token_count > 0 && token_count <=
            physical_blocks->size() * cache->block_size();
}

bool CudaDeepSeek2PagedMlaBatchView::valid() const {
    return cache != nullptr && cache->valid() && physical_blocks != nullptr &&
        !physical_blocks->empty() && layer < cache->layer_count() &&
        token_count > 0 && physical_blocks->size() <=
            std::numeric_limits<std::size_t>::max() / cache->block_size() &&
        token_count <= physical_blocks->size() * cache->block_size();
}

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
    std::size_t position,
    const CudaDeepSeek2PagedMlaView* paged) {
    const bool uses_paged = paged != nullptr;
    if (context == nullptr || hidden == nullptr || !weights.valid() ||
        (!uses_paged && position != cache_.size_tokens()) ||
        (uses_paged && (!paged->valid() || paged->token_count != position + 1)) ||
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
    if (status.is_ok() && uses_paged) status = context->write_paged_mla(
        paged->cache, paged->layer, paged->physical_block, paged->slot,
        normalized_latent_, key_rope_);
    if (status.is_ok() && !uses_paged) status = context->append_mla(
        &cache_, normalized_latent_, key_rope_);
    if (status.is_ok() && uses_paged) status = context->paged_mla_attention(
        query_nope_, query_rope_, *weights.kv_b_dequantized, *paged->cache,
        paged->layer, *paged->physical_blocks, paged->token_count,
        &attention_probabilities_, &attention_value_, config_.attention_scale);
    if (status.is_ok() && !uses_paged) status = context->mla_attention(
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
    if (!status.is_ok() && !uses_paged) cache_.trim(position);
    return status;
}

core::Status CudaDeepSeek2DenseLayer::prefill_sequence(
    cuda::CudaExecutionContext* context,
    const CudaDeepSeek2DenseLayerWeightView& weights,
    cuda::CudaTensor* hidden,
    std::size_t position_offset,
    CudaDeepSeek2DensePrefillWorkspace* workspace,
    const CudaDeepSeek2PagedMlaBatchView* paged) {
    const bool uses_paged = paged != nullptr;
    const std::size_t tokens = workspace == nullptr ? 0 : workspace->token_count_;
    if (context == nullptr || hidden == nullptr || workspace == nullptr ||
        tokens == 0 || !weights.valid() ||
        position_offset > std::numeric_limits<std::size_t>::max() - tokens ||
        (!uses_paged && position_offset != cache_.size_tokens()) ||
        (uses_paged && (!paged->valid() ||
            paged->token_count != position_offset + tokens)) ||
        hidden->contract().shape.dimensions !=
            std::vector<std::size_t>{tokens, config_.hidden_size}) {
        return core::Status::error(
            "DeepSeek2 CUDA dense Prefill input is invalid");
    }
    std::vector<std::size_t> positions(tokens);
    for (std::size_t index = 0; index < tokens; ++index) {
        positions[index] = position_offset + index;
    }
    core::Status status = context->copy(*hidden, &workspace->hidden_backup_);
    if (!status.is_ok()) return status;
    const auto rollback = [&](const core::Status& failure) {
        if (!uses_paged) {
            const core::Status trim_status = cache_.trim(position_offset);
            if (!trim_status.is_ok()) return trim_status;
        }
        const core::Status hidden_status =
            context->copy(workspace->hidden_backup_, hidden);
        return hidden_status.is_ok() ? failure : hidden_status;
    };
#define RETURN_IF_DENSE_PREFILL_ERROR(expression) do { status = (expression); \
    if (!status.is_ok()) return rollback(status); } while (false)
    RETURN_IF_DENSE_PREFILL_ERROR(context->rms_norm_batch(
        *hidden, *weights.attention_norm, &workspace->attention_norm_,
        config_.rms_norm_epsilon));
    RETURN_IF_DENSE_PREFILL_ERROR(context->linear_batch(
        workspace->attention_norm_, weights.query, &workspace->query_));
    RETURN_IF_DENSE_PREFILL_ERROR(context->linear_batch(
        workspace->attention_norm_, weights.kv_a_mqa,
        &workspace->compressed_kv_));
    RETURN_IF_DENSE_PREFILL_ERROR(context->prepare_mla_batch(
        workspace->query_, workspace->compressed_kv_,
        &workspace->query_nope_, &workspace->query_rope_,
        &workspace->latent_, &workspace->key_rope_));
    RETURN_IF_DENSE_PREFILL_ERROR(context->rms_norm_batch(
        workspace->latent_, *weights.kv_a_norm,
        &workspace->normalized_latent_, config_.rms_norm_epsilon));
    RETURN_IF_DENSE_PREFILL_ERROR(context->rope_batch(
        &workspace->query_rope_, positions, config_.rope_dimension,
        config_.rope_frequency_base, core::RotaryPairing::Adjacent,
        config_.yarn));
    RETURN_IF_DENSE_PREFILL_ERROR(context->rope_batch(
        &workspace->key_rope_, positions, config_.rope_dimension,
        config_.rope_frequency_base, core::RotaryPairing::Adjacent,
        config_.yarn));
    if (uses_paged) {
        RETURN_IF_DENSE_PREFILL_ERROR(context->write_paged_mla_batch(
            paged->cache, paged->layer, *paged->physical_blocks,
            position_offset, workspace->normalized_latent_,
            workspace->key_rope_));
        RETURN_IF_DENSE_PREFILL_ERROR(context->paged_mla_attention_batch(
            workspace->query_nope_, workspace->query_rope_,
            *weights.kv_b_dequantized, *paged->cache, paged->layer,
            *paged->physical_blocks, paged->token_count, position_offset,
            &workspace->attention_probabilities_,
            &workspace->attention_value_, config_.attention_scale));
    } else {
        RETURN_IF_DENSE_PREFILL_ERROR(context->append_mla_batch(
            &cache_, workspace->normalized_latent_, workspace->key_rope_));
        RETURN_IF_DENSE_PREFILL_ERROR(context->mla_attention_batch(
            workspace->query_nope_, workspace->query_rope_,
            *weights.kv_b_dequantized, cache_, position_offset,
            &workspace->attention_probabilities_,
            &workspace->attention_value_, config_.attention_scale));
    }
    RETURN_IF_DENSE_PREFILL_ERROR(context->linear_batch(
        workspace->attention_value_, weights.attention_output,
        &workspace->attention_output_));
    RETURN_IF_DENSE_PREFILL_ERROR(context->vector_add(
        *hidden, workspace->attention_output_,
        &workspace->attention_residual_));
    RETURN_IF_DENSE_PREFILL_ERROR(context->rms_norm_batch(
        workspace->attention_residual_, *weights.ffn_norm,
        &workspace->ffn_norm_, config_.rms_norm_epsilon));
    RETURN_IF_DENSE_PREFILL_ERROR(context->linear_batch(
        workspace->ffn_norm_, weights.ffn_gate, &workspace->gate_));
    RETURN_IF_DENSE_PREFILL_ERROR(context->linear_batch(
        workspace->ffn_norm_, weights.ffn_up, &workspace->up_));
    RETURN_IF_DENSE_PREFILL_ERROR(context->swiglu(
        workspace->gate_, workspace->up_, &workspace->activated_));
    RETURN_IF_DENSE_PREFILL_ERROR(context->linear_batch(
        workspace->activated_, weights.ffn_down, &workspace->ffn_output_));
    RETURN_IF_DENSE_PREFILL_ERROR(context->vector_add(
        workspace->attention_residual_, workspace->ffn_output_, hidden));
#undef RETURN_IF_DENSE_PREFILL_ERROR
    return core::Status::ok();
}

core::Status CudaDeepSeek2DensePrefillWorkspace::create(
    const CudaDeepSeek2DenseLayerConfig& config,
    std::size_t token_count,
    std::size_t capacity_tokens,
    CudaDeepSeek2DensePrefillWorkspace* output) {
    if (output == nullptr || token_count == 0 ||
        capacity_tokens < token_count || !config.validate().is_ok()) {
        return core::Status::error(
            "DeepSeek2 CUDA dense Prefill workspace is invalid");
    }
    CudaDeepSeek2DensePrefillWorkspace replacement;
    replacement.token_count_ = token_count;
    core::Status status = core::Status::ok();
    const auto create = [&status](cuda::CudaTensor* tensor,
                                  std::vector<std::size_t> shape) {
        if (status.is_ok()) status = make_tensor(shape, tensor);
    };
    create(&replacement.hidden_backup_, {token_count, config.hidden_size});
    create(&replacement.attention_norm_, {token_count, config.hidden_size});
    create(&replacement.query_, {token_count, config.head_count,
        config.nope_dimension + config.rope_dimension});
    create(&replacement.compressed_kv_, {token_count,
        config.latent_dimension + config.rope_dimension});
    create(&replacement.query_nope_, {token_count, config.head_count,
        config.nope_dimension});
    create(&replacement.query_rope_, {token_count, config.head_count,
        config.rope_dimension});
    create(&replacement.latent_, {token_count, config.latent_dimension});
    create(&replacement.normalized_latent_,
        {token_count, config.latent_dimension});
    create(&replacement.key_rope_,
        {token_count, 1, config.rope_dimension});
    create(&replacement.attention_probabilities_,
        {token_count, config.head_count, capacity_tokens});
    create(&replacement.attention_value_, {token_count, config.head_count,
        config.value_dimension});
    create(&replacement.attention_output_, {token_count, config.hidden_size});
    create(&replacement.attention_residual_,
        {token_count, config.hidden_size});
    create(&replacement.ffn_norm_, {token_count, config.hidden_size});
    create(&replacement.gate_, {token_count, config.ffn_dimension});
    create(&replacement.up_, {token_count, config.ffn_dimension});
    create(&replacement.activated_, {token_count, config.ffn_dimension});
    create(&replacement.ffn_output_, {token_count, config.hidden_size});
    replacement.moe_input_rows_.resize(token_count);
    replacement.moe_output_rows_.resize(token_count);
    for (std::size_t index = 0; index < token_count; ++index) {
        create(&replacement.moe_input_rows_[index], {config.hidden_size});
        create(&replacement.moe_output_rows_[index], {config.hidden_size});
    }
    if (!status.is_ok()) return status;
    *output = std::move(replacement);
    return core::Status::ok();
}

std::size_t CudaDeepSeek2DensePrefillWorkspace::token_count() const {
    return token_count_;
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
