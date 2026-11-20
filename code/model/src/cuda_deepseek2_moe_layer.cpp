// DeepSeek2 CUDA Model：连接压缩 MLA、路由专家和残差。
#include "cpp_llm_engine/model/cuda_deepseek2_moe_layer.h"

#include "cpp_llm_engine/cuda/cuda_execution_context.h"

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

core::Status CudaDeepSeek2MoeLayerConfig::validate() const {
    core::Status status = attention.validate();
    if (!status.is_ok()) return status;
    status = moe.validate();
    if (!status.is_ok()) return status;
    if (attention.hidden_size != moe.hidden_size) {
        return core::Status::error(
            "DeepSeek2 CUDA MoE layer hidden sizes differ");
    }
    return core::Status::ok();
}

core::Status CudaDeepSeek2MoeLayer::create(
    const CudaDeepSeek2MoeLayerConfig& config,
    std::size_t capacity_tokens,
    CudaDeepSeek2MoeLayer* output) {
    if (output == nullptr || !config.validate().is_ok()) {
        return core::Status::error(
            "DeepSeek2 CUDA MoE layer output/config is invalid");
    }
    CudaDeepSeek2MoeLayer replacement;
    replacement.config_ = config;
    core::Status status = cuda::CudaMlaCache::create(
        capacity_tokens, config.attention.latent_dimension,
        config.attention.rope_dimension, &replacement.cache_);
    if (status.is_ok()) status = CudaDeepSeek2MoeFeedForward::create(
        config.moe, &replacement.moe_);
    const auto create = [&status](cuda::CudaTensor* tensor,
                                  std::vector<std::size_t> shape) {
        if (status.is_ok()) status = make_tensor(shape, tensor);
    };
    create(&replacement.attention_norm_, {config.attention.hidden_size});
    create(&replacement.query_, {config.attention.head_count,
        config.attention.nope_dimension + config.attention.rope_dimension});
    create(&replacement.compressed_kv_,
        {config.attention.latent_dimension + config.attention.rope_dimension});
    create(&replacement.query_nope_,
        {config.attention.head_count, config.attention.nope_dimension});
    create(&replacement.query_rope_,
        {1, config.attention.head_count, config.attention.rope_dimension});
    create(&replacement.latent_, {config.attention.latent_dimension});
    create(&replacement.normalized_latent_,
        {config.attention.latent_dimension});
    create(&replacement.key_rope_,
        {1, 1, config.attention.rope_dimension});
    create(&replacement.attention_probabilities_,
        {config.attention.head_count, capacity_tokens});
    create(&replacement.attention_value_,
        {config.attention.head_count, config.attention.value_dimension});
    create(&replacement.attention_output_, {config.attention.hidden_size});
    create(&replacement.attention_residual_, {config.attention.hidden_size});
    create(&replacement.ffn_norm_, {config.attention.hidden_size});
    create(&replacement.ffn_output_, {config.attention.hidden_size});
    if (!status.is_ok()) return status;
    *output = std::move(replacement);
    return core::Status::ok();
}

core::Status CudaDeepSeek2MoeLayer::forward(
    cuda::CudaExecutionContext* context,
    const CudaDeepSeek2MoeWeights& weights,
    cuda::CudaTensor* hidden,
    std::size_t position,
    CudaDeepSeek2MoeRoute* route,
    const CudaDeepSeek2PagedMlaView* paged) {
    if (route == nullptr) {
        return core::Status::error(
            "DeepSeek2 CUDA MoE layer route output is null");
    }
    core::Status status = forward_attention(
        context, weights, hidden, position, paged);
    if (!status.is_ok()) return status;
    CudaDeepSeek2MoeRoute built_route;
    status = moe_.forward(
        context, weights, ffn_norm_, &ffn_output_, &built_route);
    if (status.is_ok()) status = finish_moe(context, hidden);
    if (!status.is_ok()) {
        if (paged == nullptr) cache_.trim(position);
        return status;
    }
    *route = std::move(built_route);
    return core::Status::ok();
}

core::Status CudaDeepSeek2MoeLayer::forward_attention(
    cuda::CudaExecutionContext* context,
    const CudaDeepSeek2MoeWeights& weights,
    cuda::CudaTensor* hidden,
    std::size_t position,
    const CudaDeepSeek2PagedMlaView* paged) {
    const bool uses_paged = paged != nullptr;
    if (context == nullptr || hidden == nullptr ||
        !weights.valid() || (!uses_paged && position != cache_.size_tokens()) ||
        (uses_paged && (!paged->valid() || paged->token_count != position + 1)) ||
        hidden->contract().shape.dimensions !=
            std::vector<std::size_t>{config_.attention.hidden_size}) {
        return core::Status::error(
            "DeepSeek2 CUDA MoE layer input is invalid");
    }
    core::Status status = context->rms_norm(
        *hidden, weights.attention_norm(), &attention_norm_,
        config_.attention.rms_norm_epsilon);
    if (status.is_ok()) status = context->linear(
        attention_norm_, weights.query(), &query_);
    if (status.is_ok()) status = context->linear(
        attention_norm_, weights.kv_a_mqa(), &compressed_kv_);
    if (status.is_ok()) status = context->prepare_mla(
        query_, compressed_kv_, &query_nope_, &query_rope_, &latent_,
        &key_rope_);
    if (status.is_ok()) status = context->rms_norm(
        latent_, weights.kv_a_norm(), &normalized_latent_,
        config_.attention.rms_norm_epsilon);
    if (status.is_ok()) status = context->rope(
        &query_rope_, position, config_.attention.rope_dimension,
        config_.attention.rope_frequency_base, core::RotaryPairing::Adjacent,
        config_.attention.yarn);
    if (status.is_ok()) status = context->rope(
        &key_rope_, position, config_.attention.rope_dimension,
        config_.attention.rope_frequency_base, core::RotaryPairing::Adjacent,
        config_.attention.yarn);
    if (status.is_ok() && uses_paged) status = context->write_paged_mla(
        paged->cache, paged->layer, paged->physical_block, paged->slot,
        normalized_latent_, key_rope_);
    if (status.is_ok() && !uses_paged) status = context->append_mla(
        &cache_, normalized_latent_, key_rope_);
    if (status.is_ok() && uses_paged) status = context->paged_mla_attention(
        query_nope_, query_rope_, weights.kv_b_dequantized(), *paged->cache,
        paged->layer, *paged->physical_blocks, paged->token_count,
        &attention_probabilities_, &attention_value_,
        config_.attention.attention_scale,
        config_.attention.mla_compute_path);
    if (status.is_ok() && !uses_paged) status = context->mla_attention(
        query_nope_, query_rope_, weights.kv_b_dequantized(), cache_,
        &attention_probabilities_, &attention_value_,
        config_.attention.attention_scale,
        config_.attention.mla_compute_path);
    if (status.is_ok()) status = context->linear(
        attention_value_, weights.attention_output(), &attention_output_);
    if (status.is_ok()) status = context->vector_add(
        *hidden, attention_output_, &attention_residual_);
    if (status.is_ok()) status = context->rms_norm(
        attention_residual_, weights.ffn_norm(), &ffn_norm_,
        config_.attention.rms_norm_epsilon);
    if (!status.is_ok()) {
        if (!uses_paged) cache_.trim(position);
        return status;
    }
    return core::Status::ok();
}

core::Status CudaDeepSeek2MoeLayer::prefill_sequence(
    cuda::CudaExecutionContext* context,
    const CudaDeepSeek2MoeWeights& weights,
    cuda::CudaTensor* hidden,
    std::size_t position_offset,
    CudaDeepSeek2DensePrefillWorkspace* workspace,
    CudaDeepSeek2MoeBatchWorkspace* moe_workspace,
    std::vector<CudaDeepSeek2MoeRoute>* routes,
    const CudaDeepSeek2PagedMlaBatchView* paged) {
    const bool uses_paged = paged != nullptr;
    const std::size_t tokens = workspace == nullptr ? 0 : workspace->token_count_;
    if (context == nullptr || hidden == nullptr || workspace == nullptr ||
        moe_workspace == nullptr || routes == nullptr || tokens == 0 ||
        moe_workspace->batch_size() != tokens || !weights.valid() ||
        position_offset > std::numeric_limits<std::size_t>::max() - tokens ||
        (!uses_paged && position_offset != cache_.size_tokens()) ||
        (uses_paged && (!paged->valid() ||
            paged->token_count != position_offset + tokens)) ||
        hidden->contract().shape.dimensions !=
            std::vector<std::size_t>{tokens, config_.attention.hidden_size}) {
        return core::Status::error(
            "DeepSeek2 CUDA MoE Prefill input is invalid");
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
#define RETURN_IF_MOE_PREFILL_ERROR(expression) do { status = (expression); \
    if (!status.is_ok()) return rollback(status); } while (false)
    RETURN_IF_MOE_PREFILL_ERROR(context->rms_norm_batch(
        *hidden, weights.attention_norm(), &workspace->attention_norm_,
        config_.attention.rms_norm_epsilon));
    RETURN_IF_MOE_PREFILL_ERROR(context->linear_batch(
        workspace->attention_norm_, weights.query(), &workspace->query_));
    RETURN_IF_MOE_PREFILL_ERROR(context->linear_batch(
        workspace->attention_norm_, weights.kv_a_mqa(),
        &workspace->compressed_kv_));
    RETURN_IF_MOE_PREFILL_ERROR(context->prepare_mla_batch(
        workspace->query_, workspace->compressed_kv_,
        &workspace->query_nope_, &workspace->query_rope_,
        &workspace->latent_, &workspace->key_rope_));
    RETURN_IF_MOE_PREFILL_ERROR(context->rms_norm_batch(
        workspace->latent_, weights.kv_a_norm(),
        &workspace->normalized_latent_, config_.attention.rms_norm_epsilon));
    RETURN_IF_MOE_PREFILL_ERROR(context->rope_batch(
        &workspace->query_rope_, positions, config_.attention.rope_dimension,
        config_.attention.rope_frequency_base, core::RotaryPairing::Adjacent,
        config_.attention.yarn));
    RETURN_IF_MOE_PREFILL_ERROR(context->rope_batch(
        &workspace->key_rope_, positions, config_.attention.rope_dimension,
        config_.attention.rope_frequency_base, core::RotaryPairing::Adjacent,
        config_.attention.yarn));
    if (uses_paged) {
        RETURN_IF_MOE_PREFILL_ERROR(context->write_paged_mla_batch(
            paged->cache, paged->layer, *paged->physical_blocks,
            position_offset, workspace->normalized_latent_,
            workspace->key_rope_));
        RETURN_IF_MOE_PREFILL_ERROR(context->paged_mla_attention_batch(
            workspace->query_nope_, workspace->query_rope_,
            weights.kv_b_dequantized(), *paged->cache, paged->layer,
            *paged->physical_blocks, paged->token_count, position_offset,
            &workspace->attention_probabilities_,
            &workspace->attention_value_, config_.attention.attention_scale,
            config_.attention.mla_compute_path));
    } else {
        RETURN_IF_MOE_PREFILL_ERROR(context->append_mla_batch(
            &cache_, workspace->normalized_latent_, workspace->key_rope_));
        RETURN_IF_MOE_PREFILL_ERROR(context->mla_attention_batch(
            workspace->query_nope_, workspace->query_rope_,
            weights.kv_b_dequantized(), cache_, position_offset,
            &workspace->attention_probabilities_,
            &workspace->attention_value_, config_.attention.attention_scale,
            config_.attention.mla_compute_path));
    }
    RETURN_IF_MOE_PREFILL_ERROR(context->linear_batch(
        workspace->attention_value_, weights.attention_output(),
        &workspace->attention_output_));
    RETURN_IF_MOE_PREFILL_ERROR(context->vector_add(
        *hidden, workspace->attention_output_,
        &workspace->attention_residual_));
    RETURN_IF_MOE_PREFILL_ERROR(context->rms_norm_batch(
        workspace->attention_residual_, weights.ffn_norm(),
        &workspace->ffn_norm_, config_.attention.rms_norm_epsilon));
    std::vector<cuda::CudaTensor*> input_rows;
    std::vector<const cuda::CudaTensor*> normalized_rows;
    std::vector<cuda::CudaTensor*> output_rows;
    input_rows.reserve(tokens);
    normalized_rows.reserve(tokens);
    output_rows.reserve(tokens);
    for (std::size_t index = 0; index < tokens; ++index) {
        input_rows.push_back(&workspace->moe_input_rows_[index]);
        normalized_rows.push_back(&workspace->moe_input_rows_[index]);
        output_rows.push_back(&workspace->moe_output_rows_[index]);
    }
    RETURN_IF_MOE_PREFILL_ERROR(context->unpack_rows(
        workspace->ffn_norm_, input_rows));
    RETURN_IF_MOE_PREFILL_ERROR(moe_workspace->forward(
        context, weights, normalized_rows, output_rows, routes));
    std::vector<const cuda::CudaTensor*> packed_rows;
    packed_rows.reserve(tokens);
    for (const auto& row : workspace->moe_output_rows_) {
        packed_rows.push_back(&row);
    }
    RETURN_IF_MOE_PREFILL_ERROR(context->pack_rows(
        packed_rows, &workspace->ffn_output_));
    RETURN_IF_MOE_PREFILL_ERROR(context->vector_add(
        workspace->attention_residual_, workspace->ffn_output_, hidden));
#undef RETURN_IF_MOE_PREFILL_ERROR
    return core::Status::ok();
}

core::Status CudaDeepSeek2MoeLayer::finish_moe(
    cuda::CudaExecutionContext* context,
    cuda::CudaTensor* hidden) {
    if (context == nullptr || hidden == nullptr) {
        return core::Status::error(
            "DeepSeek2 CUDA MoE finish input is invalid");
    }
    return context->vector_add(attention_residual_, ffn_output_, hidden);
}

const cuda::CudaTensor& CudaDeepSeek2MoeLayer::normalized_ffn_input() const {
    return ffn_norm_;
}
cuda::CudaTensor* CudaDeepSeek2MoeLayer::moe_output() {
    return &ffn_output_;
}

void CudaDeepSeek2MoeLayer::reset() { cache_.clear(); }
void CudaDeepSeek2MoeLayer::trim(std::size_t position) {
    cache_.trim(position);
}
std::size_t CudaDeepSeek2MoeLayer::position() const {
    return cache_.size_tokens();
}
const cuda::CudaMlaCache& CudaDeepSeek2MoeLayer::cache() const {
    return cache_;
}

}  // namespace cpp_llm_engine::model
