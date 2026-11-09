// CUDA Decoder：实现单 token Llama Decoder 层的可解释组装。
// 职责：复用 ExecutionContext，不在层内实现数学 kernel 或拥有请求 KV。
// 契约：forward 成功后 hidden 原地更新，cache 增加一个 token。
// 验证：cpp_llm_engine_cuda_decoder_layer_test。
#include "cpp_llm_engine/cuda/cuda_decoder_layer.h"

#include "cpp_llm_engine/cuda/cuda_execution_context.h"
#include "cpp_llm_engine/cuda/cuda_kv_cache.h"
#include "cpp_llm_engine/cuda/cuda_paged_kv_cache.h"

#include <cmath>
#include <limits>
#include <utility>
#include <vector>

namespace cpp_llm_engine::cuda {
namespace {

bool tensor_matches(const CudaTensor* tensor, std::size_t width) {
    return tensor != nullptr && !tensor->empty() &&
        tensor->contract().dtype == core::DType::Float32 &&
        tensor->contract().device == core::DeviceKind::CUDA &&
        tensor->contract().shape.dimensions == std::vector<std::size_t>{width};
}

bool linear_weight_matches(
    const CudaLinearWeight& weight,
    std::size_t input_width,
    std::size_t output_width) {
    if (!weight.valid()) return false;
    if (weight.kind() == CudaLinearWeight::Kind::Float32) {
        const CudaTensor* tensor = weight.float32_tensor();
        return tensor != nullptr &&
            tensor->contract().dtype == core::DType::Float32 &&
            tensor->contract().device == core::DeviceKind::CUDA &&
            tensor->contract().shape.dimensions ==
                std::vector<std::size_t>{input_width, output_width};
    }
    const CudaQuantizedMatrix* matrix = weight.quantized_matrix();
    return matrix != nullptr && matrix->cols() == input_width &&
        matrix->rows() == output_width;
}

bool qk_norm_weights_match(
    const CudaDecoderLayerWeights& weights,
    std::size_t head_dimension) {
    if (weights.query_norm == nullptr && weights.key_norm == nullptr) return true;
    return tensor_matches(weights.query_norm, head_dimension) &&
        tensor_matches(weights.key_norm, head_dimension);
}

}  // namespace

core::Status CudaDecoderLayer::create(
    const CudaDecoderLayerConfig& config, CudaDecoderLayer* output) {
    if (output == nullptr || config.hidden_size == 0 || config.ffn_size == 0 ||
        config.query_heads == 0 || config.kv_heads == 0 ||
        config.query_heads % config.kv_heads != 0 || config.head_dimension == 0 ||
        config.query_heads > std::numeric_limits<std::size_t>::max() /
                                 config.head_dimension ||
        config.rotary_dimension == 0 || config.rotary_dimension > config.head_dimension ||
        config.rotary_dimension % 2 != 0 || config.rms_epsilon <= 0.0F ||
        !std::isfinite(config.rms_epsilon) || config.rope_frequency_base <= 0.0F ||
        (config.rotary_pairing != core::RotaryPairing::Adjacent &&
         config.rotary_pairing != core::RotaryPairing::SplitHalf)) {
        return core::Status::error("CudaDecoderLayer config is invalid");
    }
    CudaDecoderLayer replacement;
    replacement.config_ = config;
    const std::vector<std::vector<std::size_t>> shapes{
        {config.hidden_size}, {config.hidden_size},
        {1, config.query_heads, config.head_dimension},
        {1, config.kv_heads, config.head_dimension}, {1, config.kv_heads, config.head_dimension},
        {1, config.query_heads, config.head_dimension}, {config.hidden_size},
        {config.hidden_size}, {config.ffn_size}, {config.ffn_size},
        {config.ffn_size}, {config.hidden_size}};
    CudaTensor* tensors[]{
        &replacement.hidden_backup_, &replacement.attention_normalized_,
        &replacement.query_, &replacement.key_,
        &replacement.value_, &replacement.attention_context_, &replacement.projection_,
        &replacement.ffn_normalized_, &replacement.gate_, &replacement.up_,
        &replacement.swiglu_, &replacement.down_};
    for (std::size_t index = 0; index < shapes.size(); ++index) {
        const core::Status status = CudaTensor::create_contiguous(
            shapes[index], core::DType::Float32, tensors[index]);
        if (!status.is_ok()) return status;
    }
    *output = std::move(replacement);
    return core::Status::ok();
}

core::Status CudaDecoderLayer::forward(
    CudaExecutionContext* context, const CudaDecoderLayerWeights& weights,
    CudaKvCache* cache, CudaTensor* hidden, std::size_t position) {
    if (context == nullptr || cache == nullptr || hidden == nullptr) {
        return core::Status::error("CudaDecoderLayer forward input is null");
    }
    const std::size_t kv_width = config_.kv_heads * config_.head_dimension;
    const std::size_t query_width = config_.query_heads * config_.head_dimension;
    if (!tensor_matches(hidden, config_.hidden_size) ||
        !tensor_matches(weights.attention_norm, config_.hidden_size) ||
        !tensor_matches(weights.ffn_norm, config_.hidden_size) ||
        !qk_norm_weights_match(weights, config_.head_dimension) ||
        !linear_weight_matches(
            weights.query, config_.hidden_size, query_width) ||
        !linear_weight_matches(weights.key, config_.hidden_size, kv_width) ||
        !linear_weight_matches(weights.value, config_.hidden_size, kv_width) ||
        !linear_weight_matches(
            weights.attention_output, query_width, config_.hidden_size) ||
        !linear_weight_matches(weights.gate, config_.hidden_size, config_.ffn_size) ||
        !linear_weight_matches(weights.up, config_.hidden_size, config_.ffn_size) ||
        !linear_weight_matches(weights.down, config_.ffn_size, config_.hidden_size)) {
        return core::Status::error("CudaDecoderLayer tensor or weight shape is invalid");
    }
    if (!cache->valid() || cache->kv_head_count() != config_.kv_heads ||
        cache->head_dimension() != config_.head_dimension ||
        position != cache->size_tokens()) {
        return core::Status::error("CudaDecoderLayer cache position or shape is invalid");
    }
    const std::size_t initial_cache_size = cache->size_tokens();
    core::Status status = context->copy(*hidden, &hidden_backup_);
    if (!status.is_ok()) return status;
    const auto rollback = [&](const core::Status& failure) {
        const core::Status cache_status = cache->trim(initial_cache_size);
        const core::Status hidden_status = context->copy(hidden_backup_, hidden);
        if (!cache_status.is_ok()) return cache_status;
        if (!hidden_status.is_ok()) return hidden_status;
        return failure;
    };
#define RETURN_IF_ERROR(expression) do { status = (expression); \
    if (!status.is_ok()) return rollback(status); } while (false)
    RETURN_IF_ERROR(context->rms_norm(
        *hidden, *weights.attention_norm, &attention_normalized_, config_.rms_epsilon));
    RETURN_IF_ERROR(context->linear(attention_normalized_, weights.query, &query_));
    RETURN_IF_ERROR(context->linear(attention_normalized_, weights.key, &key_));
    RETURN_IF_ERROR(context->linear(attention_normalized_, weights.value, &value_));
    if (weights.query_norm != nullptr) {
        RETURN_IF_ERROR(context->rms_norm_batch(
            query_, *weights.query_norm, &query_, config_.rms_epsilon));
        RETURN_IF_ERROR(context->rms_norm_batch(
            key_, *weights.key_norm, &key_, config_.rms_epsilon));
    }
    RETURN_IF_ERROR(context->rope(
        &query_, position, config_.rotary_dimension, config_.rope_frequency_base,
        config_.rotary_pairing));
    RETURN_IF_ERROR(context->rope(
        &key_, position, config_.rotary_dimension, config_.rope_frequency_base,
        config_.rotary_pairing));
    RETURN_IF_ERROR(context->append_kv(cache, key_, value_));
    RETURN_IF_ERROR(context->gqa_attention(
        query_, *cache, &attention_context_, position));
    RETURN_IF_ERROR(context->linear(
        attention_context_, weights.attention_output, &projection_));
    RETURN_IF_ERROR(context->vector_add(*hidden, projection_, hidden));
    RETURN_IF_ERROR(context->rms_norm(
        *hidden, *weights.ffn_norm, &ffn_normalized_, config_.rms_epsilon));
    RETURN_IF_ERROR(context->linear(ffn_normalized_, weights.gate, &gate_));
    RETURN_IF_ERROR(context->linear(ffn_normalized_, weights.up, &up_));
    RETURN_IF_ERROR(context->swiglu(gate_, up_, &swiglu_));
    RETURN_IF_ERROR(context->linear(swiglu_, weights.down, &down_));
    RETURN_IF_ERROR(context->vector_add(*hidden, down_, hidden));
#undef RETURN_IF_ERROR
    return core::Status::ok();
}

core::Status CudaDecoderLayer::forward_paged(
    CudaExecutionContext* context,
    const CudaDecoderLayerWeights& weights,
    CudaPagedKvCache* cache,
    std::size_t layer,
    const std::vector<std::size_t>& physical_blocks,
    CudaTensor* hidden,
    std::size_t position) {
    if (context == nullptr || cache == nullptr || hidden == nullptr ||
        !cache->valid() || layer >= cache->layer_count() ||
        cache->kv_head_count() != config_.kv_heads ||
        cache->head_dimension() != config_.head_dimension ||
        physical_blocks.empty() || position / cache->block_size() >= physical_blocks.size()) {
        return core::Status::error("CudaDecoderLayer paged input is invalid");
    }
    const std::size_t kv_width = config_.kv_heads * config_.head_dimension;
    const std::size_t query_width = config_.query_heads * config_.head_dimension;
    if (!tensor_matches(hidden, config_.hidden_size) ||
        !tensor_matches(weights.attention_norm, config_.hidden_size) ||
        !tensor_matches(weights.ffn_norm, config_.hidden_size) ||
        !qk_norm_weights_match(weights, config_.head_dimension) ||
        !linear_weight_matches(weights.query, config_.hidden_size, query_width) ||
        !linear_weight_matches(weights.key, config_.hidden_size, kv_width) ||
        !linear_weight_matches(weights.value, config_.hidden_size, kv_width) ||
        !linear_weight_matches(
            weights.attention_output, query_width, config_.hidden_size) ||
        !linear_weight_matches(weights.gate, config_.hidden_size, config_.ffn_size) ||
        !linear_weight_matches(weights.up, config_.hidden_size, config_.ffn_size) ||
        !linear_weight_matches(weights.down, config_.ffn_size, config_.hidden_size)) {
        return core::Status::error("CudaDecoderLayer paged tensor or weight shape is invalid");
    }
    core::Status status = context->copy(*hidden, &hidden_backup_);
    if (!status.is_ok()) return status;
    const auto rollback = [&](const core::Status& failure) {
        const core::Status hidden_status = context->copy(hidden_backup_, hidden);
        return hidden_status.is_ok() ? failure : hidden_status;
    };
#define RETURN_IF_PAGED_ERROR(expression) do { status = (expression); \
    if (!status.is_ok()) return rollback(status); } while (false)
    RETURN_IF_PAGED_ERROR(context->rms_norm(
        *hidden, *weights.attention_norm, &attention_normalized_, config_.rms_epsilon));
    RETURN_IF_PAGED_ERROR(context->linear(attention_normalized_, weights.query, &query_));
    RETURN_IF_PAGED_ERROR(context->linear(attention_normalized_, weights.key, &key_));
    RETURN_IF_PAGED_ERROR(context->linear(attention_normalized_, weights.value, &value_));
    if (weights.query_norm != nullptr) {
        RETURN_IF_PAGED_ERROR(context->rms_norm_batch(
            query_, *weights.query_norm, &query_, config_.rms_epsilon));
        RETURN_IF_PAGED_ERROR(context->rms_norm_batch(
            key_, *weights.key_norm, &key_, config_.rms_epsilon));
    }
    RETURN_IF_PAGED_ERROR(context->rope(
        &query_, position, config_.rotary_dimension, config_.rope_frequency_base,
        config_.rotary_pairing));
    RETURN_IF_PAGED_ERROR(context->rope(
        &key_, position, config_.rotary_dimension, config_.rope_frequency_base,
        config_.rotary_pairing));
    const std::size_t physical_block = physical_blocks[position / cache->block_size()];
    const std::size_t slot = position % cache->block_size();
    RETURN_IF_PAGED_ERROR(context->write_paged_kv(
        cache, layer, physical_block, slot, key_, value_));
    RETURN_IF_PAGED_ERROR(context->paged_gqa_attention(
        query_, *cache, layer, physical_blocks, position + 1,
        &attention_context_, position));
    RETURN_IF_PAGED_ERROR(context->linear(
        attention_context_, weights.attention_output, &projection_));
    RETURN_IF_PAGED_ERROR(context->vector_add(*hidden, projection_, hidden));
    RETURN_IF_PAGED_ERROR(context->rms_norm(
        *hidden, *weights.ffn_norm, &ffn_normalized_, config_.rms_epsilon));
    RETURN_IF_PAGED_ERROR(context->linear(ffn_normalized_, weights.gate, &gate_));
    RETURN_IF_PAGED_ERROR(context->linear(ffn_normalized_, weights.up, &up_));
    RETURN_IF_PAGED_ERROR(context->swiglu(gate_, up_, &swiglu_));
    RETURN_IF_PAGED_ERROR(context->linear(swiglu_, weights.down, &down_));
    RETURN_IF_PAGED_ERROR(context->vector_add(*hidden, down_, hidden));
#undef RETURN_IF_PAGED_ERROR
    return core::Status::ok();
}

const CudaTensor& CudaDecoderLayer::attention_normalized() const { return attention_normalized_; }
const CudaTensor& CudaDecoderLayer::attention_context() const { return attention_context_; }
const CudaTensor& CudaDecoderLayer::ffn_normalized() const { return ffn_normalized_; }
const CudaTensor& CudaDecoderLayer::swiglu_output() const { return swiglu_; }

}  // namespace cpp_llm_engine::cuda
