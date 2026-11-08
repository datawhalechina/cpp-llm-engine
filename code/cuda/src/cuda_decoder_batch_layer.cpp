// CUDA Decoder Batch：实现多请求 Decode 与单序列 Tensor Prefill 的事务编排。
// 契约：成功时按执行模式推进 K/V；失败时 cache 与 hidden 恢复。
// 验证：cpp_llm_engine_cuda_decoder_layer_test。
#include "cpp_llm_engine/cuda/cuda_decoder_batch_layer.h"

#include "cpp_llm_engine/cuda/cuda_execution_context.h"
#include "cpp_llm_engine/cuda/cuda_kv_cache.h"
#include "cpp_llm_engine/cuda/cuda_paged_kv_cache.h"

#include <limits>
#include <unordered_set>
#include <utility>

namespace cpp_llm_engine::cuda {

core::Status CudaDecoderBatchLayer::create(
    const CudaDecoderLayerConfig& config,
    std::size_t batch_size,
    CudaDecoderBatchLayer* output) {
    if (output == nullptr || batch_size == 0) {
        return core::Status::error("CUDA batch layer output or batch size is invalid");
    }
    CudaDecoderLayer validation;
    core::Status status = CudaDecoderLayer::create(config, &validation);
    if (!status.is_ok()) return status;

    CudaDecoderBatchLayer replacement;
    replacement.config_ = config;
    replacement.batch_size_ = batch_size;
    const std::vector<std::vector<std::size_t>> shapes{
        {batch_size, config.hidden_size}, {batch_size, config.hidden_size},
        {batch_size, config.query_heads, config.head_dimension},
        {batch_size, config.kv_heads, config.head_dimension},
        {batch_size, config.kv_heads, config.head_dimension},
        {batch_size, config.query_heads, config.head_dimension},
        {batch_size, config.hidden_size}, {batch_size, config.hidden_size},
        {batch_size, config.ffn_size}, {batch_size, config.ffn_size},
        {batch_size, config.ffn_size}, {batch_size, config.hidden_size}};
    CudaTensor* tensors[]{
        &replacement.hidden_backup_, &replacement.attention_normalized_,
        &replacement.query_, &replacement.key_, &replacement.value_,
        &replacement.attention_context_, &replacement.projection_,
        &replacement.ffn_normalized_, &replacement.gate_, &replacement.up_,
        &replacement.swiglu_, &replacement.down_};
    for (std::size_t index = 0; index < shapes.size(); ++index) {
        status = CudaTensor::create_contiguous(
            shapes[index], core::DType::Float32, tensors[index]);
        if (!status.is_ok()) return status;
    }
    replacement.query_rows_.resize(batch_size);
    replacement.key_rows_.resize(batch_size);
    replacement.value_rows_.resize(batch_size);
    replacement.attention_rows_.resize(batch_size);
    for (std::size_t index = 0; index < batch_size; ++index) {
        for (auto item : {
                 std::pair<CudaTensor*, std::vector<std::size_t>>{
                     &replacement.query_rows_[index],
                     {1, config.query_heads, config.head_dimension}},
                 {&replacement.key_rows_[index],
                  {1, config.kv_heads, config.head_dimension}},
                 {&replacement.value_rows_[index],
                  {1, config.kv_heads, config.head_dimension}},
                 {&replacement.attention_rows_[index],
                  {1, config.query_heads, config.head_dimension}}}) {
            status = CudaTensor::create_contiguous(
                item.second, core::DType::Float32, item.first);
            if (!status.is_ok()) return status;
        }
    }
    *output = std::move(replacement);
    return core::Status::ok();
}

core::Status CudaDecoderBatchLayer::forward(
    CudaExecutionContext* context,
    const CudaDecoderLayerWeights& weights,
    const std::vector<CudaKvCache*>& caches,
    CudaTensor* hidden,
    const std::vector<std::size_t>& positions) {
    if (context == nullptr || hidden == nullptr || caches.size() != batch_size_ ||
        positions.size() != batch_size_ ||
        weights.attention_norm == nullptr || weights.ffn_norm == nullptr ||
        !weights.query.valid() || !weights.key.valid() || !weights.value.valid() ||
        !weights.attention_output.valid() || !weights.gate.valid() ||
        !weights.up.valid() || !weights.down.valid() ||
        hidden->contract().shape.dimensions !=
            std::vector<std::size_t>{batch_size_, config_.hidden_size}) {
        return core::Status::error("CUDA batch layer input shape is invalid");
    }
    std::unordered_set<CudaKvCache*> unique;
    for (std::size_t index = 0; index < batch_size_; ++index) {
        if (caches[index] == nullptr || !unique.insert(caches[index]).second ||
            !caches[index]->valid() ||
            caches[index]->kv_head_count() != config_.kv_heads ||
            caches[index]->head_dimension() != config_.head_dimension ||
            caches[index]->size_tokens() != positions[index]) {
            return core::Status::error("CUDA batch layer cache state is invalid");
        }
    }
    core::Status status = context->copy(*hidden, &hidden_backup_);
    if (!status.is_ok()) return status;
    const auto rollback = [&](const core::Status& failure) {
        for (std::size_t index = 0; index < batch_size_; ++index) {
            const core::Status trim_status = caches[index]->trim(positions[index]);
            if (!trim_status.is_ok()) return trim_status;
        }
        const core::Status hidden_status = context->copy(hidden_backup_, hidden);
        return hidden_status.is_ok() ? failure : hidden_status;
    };
#define RETURN_IF_ERROR(expression) do { status = (expression); \
    if (!status.is_ok()) return rollback(status); } while (false)
    RETURN_IF_ERROR(context->rms_norm_batch(
        *hidden, *weights.attention_norm, &attention_normalized_, config_.rms_epsilon));
    RETURN_IF_ERROR(context->linear_batch(attention_normalized_, weights.query, &query_));
    RETURN_IF_ERROR(context->linear_batch(attention_normalized_, weights.key, &key_));
    RETURN_IF_ERROR(context->linear_batch(attention_normalized_, weights.value, &value_));
    RETURN_IF_ERROR(context->rope_batch(
        &query_, positions, config_.rotary_dimension, config_.rope_frequency_base));
    RETURN_IF_ERROR(context->rope_batch(
        &key_, positions, config_.rotary_dimension, config_.rope_frequency_base));

    std::vector<CudaTensor*> query_rows;
    std::vector<CudaTensor*> key_rows;
    std::vector<CudaTensor*> value_rows;
    std::vector<const CudaTensor*> attention_rows;
    for (std::size_t index = 0; index < batch_size_; ++index) {
        query_rows.push_back(&query_rows_[index]);
        key_rows.push_back(&key_rows_[index]);
        value_rows.push_back(&value_rows_[index]);
        attention_rows.push_back(&attention_rows_[index]);
    }
    RETURN_IF_ERROR(context->unpack_rows(query_, query_rows));
    RETURN_IF_ERROR(context->unpack_rows(key_, key_rows));
    RETURN_IF_ERROR(context->unpack_rows(value_, value_rows));
    for (std::size_t index = 0; index < batch_size_; ++index) {
        RETURN_IF_ERROR(context->append_kv(
            caches[index], key_rows_[index], value_rows_[index]));
        RETURN_IF_ERROR(context->gqa_attention(
            query_rows_[index], *caches[index], &attention_rows_[index], positions[index]));
    }
    RETURN_IF_ERROR(context->pack_rows(attention_rows, &attention_context_));
    RETURN_IF_ERROR(context->linear_batch(
        attention_context_, weights.attention_output, &projection_));
    RETURN_IF_ERROR(context->vector_add(*hidden, projection_, hidden));
    RETURN_IF_ERROR(context->rms_norm_batch(
        *hidden, *weights.ffn_norm, &ffn_normalized_, config_.rms_epsilon));
    RETURN_IF_ERROR(context->linear_batch(ffn_normalized_, weights.gate, &gate_));
    RETURN_IF_ERROR(context->linear_batch(ffn_normalized_, weights.up, &up_));
    RETURN_IF_ERROR(context->swiglu(gate_, up_, &swiglu_));
    RETURN_IF_ERROR(context->linear_batch(swiglu_, weights.down, &down_));
    RETURN_IF_ERROR(context->vector_add(*hidden, down_, hidden));
#undef RETURN_IF_ERROR
    return core::Status::ok();
}

core::Status CudaDecoderBatchLayer::prefill_sequence(
    CudaExecutionContext* context,
    const CudaDecoderLayerWeights& weights,
    CudaKvCache* cache,
    CudaTensor* hidden,
    std::size_t position_offset) {
    if (context == nullptr || cache == nullptr || hidden == nullptr ||
        !cache->valid() || cache->size_tokens() != position_offset ||
        cache->kv_head_count() != config_.kv_heads ||
        cache->head_dimension() != config_.head_dimension ||
        weights.attention_norm == nullptr || weights.ffn_norm == nullptr ||
        !weights.query.valid() || !weights.key.valid() || !weights.value.valid() ||
        !weights.attention_output.valid() || !weights.gate.valid() ||
        !weights.up.valid() || !weights.down.valid() ||
        batch_size_ - 1 > std::numeric_limits<std::size_t>::max() - position_offset ||
        hidden->contract().shape.dimensions !=
            std::vector<std::size_t>{batch_size_, config_.hidden_size}) {
        return core::Status::error("CUDA sequence prefill input shape is invalid");
    }
    std::vector<std::size_t> positions(batch_size_);
    for (std::size_t index = 0; index < batch_size_; ++index) {
        positions[index] = position_offset + index;
    }
    core::Status status = context->copy(*hidden, &hidden_backup_);
    if (!status.is_ok()) return status;
    const auto rollback = [&](const core::Status& failure) {
        const core::Status trim_status = cache->trim(position_offset);
        const core::Status hidden_status = context->copy(hidden_backup_, hidden);
        if (!trim_status.is_ok()) return trim_status;
        return hidden_status.is_ok() ? failure : hidden_status;
    };
#define RETURN_IF_PREFILL_ERROR(expression) do { status = (expression); \
    if (!status.is_ok()) return rollback(status); } while (false)
    RETURN_IF_PREFILL_ERROR(context->rms_norm_batch(
        *hidden, *weights.attention_norm, &attention_normalized_, config_.rms_epsilon));
    RETURN_IF_PREFILL_ERROR(context->linear_batch(
        attention_normalized_, weights.query, &query_));
    RETURN_IF_PREFILL_ERROR(context->linear_batch(
        attention_normalized_, weights.key, &key_));
    RETURN_IF_PREFILL_ERROR(context->linear_batch(
        attention_normalized_, weights.value, &value_));
    RETURN_IF_PREFILL_ERROR(context->rope_batch(
        &query_, positions, config_.rotary_dimension, config_.rope_frequency_base));
    RETURN_IF_PREFILL_ERROR(context->rope_batch(
        &key_, positions, config_.rotary_dimension, config_.rope_frequency_base));
    RETURN_IF_PREFILL_ERROR(context->append_kv(cache, key_, value_));
    RETURN_IF_PREFILL_ERROR(context->gqa_attention(
        query_, *cache, &attention_context_, position_offset));
    RETURN_IF_PREFILL_ERROR(context->linear_batch(
        attention_context_, weights.attention_output, &projection_));
    RETURN_IF_PREFILL_ERROR(context->vector_add(*hidden, projection_, hidden));
    RETURN_IF_PREFILL_ERROR(context->rms_norm_batch(
        *hidden, *weights.ffn_norm, &ffn_normalized_, config_.rms_epsilon));
    RETURN_IF_PREFILL_ERROR(context->linear_batch(
        ffn_normalized_, weights.gate, &gate_));
    RETURN_IF_PREFILL_ERROR(context->linear_batch(
        ffn_normalized_, weights.up, &up_));
    RETURN_IF_PREFILL_ERROR(context->swiglu(gate_, up_, &swiglu_));
    RETURN_IF_PREFILL_ERROR(context->linear_batch(swiglu_, weights.down, &down_));
    RETURN_IF_PREFILL_ERROR(context->vector_add(*hidden, down_, hidden));
#undef RETURN_IF_PREFILL_ERROR
    return core::Status::ok();
}

core::Status CudaDecoderBatchLayer::forward_paged(
    CudaExecutionContext* context,
    const CudaDecoderLayerWeights& weights,
    CudaPagedKvCache* cache,
    std::size_t layer,
    const std::vector<std::vector<std::size_t>>& physical_blocks,
    CudaTensor* hidden,
    const std::vector<std::size_t>& positions) {
    if (context == nullptr || cache == nullptr || hidden == nullptr ||
        !cache->valid() || layer >= cache->layer_count() ||
        cache->kv_head_count() != config_.kv_heads ||
        cache->head_dimension() != config_.head_dimension ||
        physical_blocks.size() != batch_size_ || positions.size() != batch_size_ ||
        weights.attention_norm == nullptr || weights.ffn_norm == nullptr ||
        !weights.query.valid() || !weights.key.valid() || !weights.value.valid() ||
        !weights.attention_output.valid() || !weights.gate.valid() ||
        !weights.up.valid() || !weights.down.valid() ||
        hidden->contract().shape.dimensions !=
            std::vector<std::size_t>{batch_size_, config_.hidden_size}) {
        return core::Status::error("CUDA paged batch layer input is invalid");
    }
    std::unordered_set<std::size_t> destinations;
    for (std::size_t index = 0; index < batch_size_; ++index) {
        if (physical_blocks[index].empty() ||
            positions[index] / cache->block_size() >= physical_blocks[index].size()) {
            return core::Status::error("CUDA paged batch page table is invalid");
        }
        const std::size_t physical_block =
            physical_blocks[index][positions[index] / cache->block_size()];
        if (physical_block >= cache->total_blocks()) {
            return core::Status::error("CUDA paged batch block is out of range");
        }
        const std::size_t destination =
            physical_block * cache->block_size() + positions[index] % cache->block_size();
        if (!destinations.insert(destination).second) {
            return core::Status::error("CUDA paged batch destinations must be unique");
        }
    }

    core::Status status = context->copy(*hidden, &hidden_backup_);
    if (!status.is_ok()) return status;
    const auto rollback = [&](const core::Status& failure) {
        const core::Status hidden_status = context->copy(hidden_backup_, hidden);
        return hidden_status.is_ok() ? failure : hidden_status;
    };
#define RETURN_IF_PAGED_BATCH_ERROR(expression) do { status = (expression); \
    if (!status.is_ok()) return rollback(status); } while (false)
    RETURN_IF_PAGED_BATCH_ERROR(context->rms_norm_batch(
        *hidden, *weights.attention_norm, &attention_normalized_, config_.rms_epsilon));
    RETURN_IF_PAGED_BATCH_ERROR(context->linear_batch(
        attention_normalized_, weights.query, &query_));
    RETURN_IF_PAGED_BATCH_ERROR(context->linear_batch(
        attention_normalized_, weights.key, &key_));
    RETURN_IF_PAGED_BATCH_ERROR(context->linear_batch(
        attention_normalized_, weights.value, &value_));
    RETURN_IF_PAGED_BATCH_ERROR(context->rope_batch(
        &query_, positions, config_.rotary_dimension, config_.rope_frequency_base));
    RETURN_IF_PAGED_BATCH_ERROR(context->rope_batch(
        &key_, positions, config_.rotary_dimension, config_.rope_frequency_base));

    std::vector<CudaTensor*> query_rows;
    std::vector<CudaTensor*> key_rows;
    std::vector<CudaTensor*> value_rows;
    std::vector<const CudaTensor*> attention_rows;
    for (std::size_t index = 0; index < batch_size_; ++index) {
        query_rows.push_back(&query_rows_[index]);
        key_rows.push_back(&key_rows_[index]);
        value_rows.push_back(&value_rows_[index]);
        attention_rows.push_back(&attention_rows_[index]);
    }
    RETURN_IF_PAGED_BATCH_ERROR(context->unpack_rows(query_, query_rows));
    RETURN_IF_PAGED_BATCH_ERROR(context->unpack_rows(key_, key_rows));
    RETURN_IF_PAGED_BATCH_ERROR(context->unpack_rows(value_, value_rows));
    for (std::size_t index = 0; index < batch_size_; ++index) {
        const std::size_t physical_block =
            physical_blocks[index][positions[index] / cache->block_size()];
        const std::size_t slot = positions[index] % cache->block_size();
        RETURN_IF_PAGED_BATCH_ERROR(context->write_paged_kv(
            cache, layer, physical_block, slot,
            key_rows_[index], value_rows_[index]));
        RETURN_IF_PAGED_BATCH_ERROR(context->paged_gqa_attention(
            query_rows_[index], *cache, layer, physical_blocks[index],
            positions[index] + 1, &attention_rows_[index], positions[index]));
    }
    RETURN_IF_PAGED_BATCH_ERROR(context->pack_rows(
        attention_rows, &attention_context_));
    RETURN_IF_PAGED_BATCH_ERROR(context->linear_batch(
        attention_context_, weights.attention_output, &projection_));
    RETURN_IF_PAGED_BATCH_ERROR(context->vector_add(*hidden, projection_, hidden));
    RETURN_IF_PAGED_BATCH_ERROR(context->rms_norm_batch(
        *hidden, *weights.ffn_norm, &ffn_normalized_, config_.rms_epsilon));
    RETURN_IF_PAGED_BATCH_ERROR(context->linear_batch(
        ffn_normalized_, weights.gate, &gate_));
    RETURN_IF_PAGED_BATCH_ERROR(context->linear_batch(
        ffn_normalized_, weights.up, &up_));
    RETURN_IF_PAGED_BATCH_ERROR(context->swiglu(gate_, up_, &swiglu_));
    RETURN_IF_PAGED_BATCH_ERROR(context->linear_batch(swiglu_, weights.down, &down_));
    RETURN_IF_PAGED_BATCH_ERROR(context->vector_add(*hidden, down_, hidden));
#undef RETURN_IF_PAGED_BATCH_ERROR
    return core::Status::ok();
}

std::size_t CudaDecoderBatchLayer::device_bytes() const {
    std::size_t bytes = hidden_backup_.size_bytes() +
        attention_normalized_.size_bytes() + query_.size_bytes() +
        key_.size_bytes() + value_.size_bytes() + attention_context_.size_bytes() +
        projection_.size_bytes() + ffn_normalized_.size_bytes() +
        gate_.size_bytes() + up_.size_bytes() + swiglu_.size_bytes() +
        down_.size_bytes();
    for (std::size_t index = 0; index < query_rows_.size(); ++index) {
        bytes += query_rows_[index].size_bytes() + key_rows_[index].size_bytes() +
            value_rows_[index].size_bytes() + attention_rows_[index].size_bytes();
    }
    return bytes;
}

}  // namespace cpp_llm_engine::cuda
