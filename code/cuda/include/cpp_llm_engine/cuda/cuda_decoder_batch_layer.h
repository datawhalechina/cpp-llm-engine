// CUDA Decoder Batch：声明多请求 Decode 与单序列多 token Prefill 层。
// 职责：复用批量算子，并区分独立请求 K/V 和单序列因果 K/V。
// 验证：合成双请求及 2-token Prefill 与顺序 Decoder 的结果对照。
#pragma once

#include "cpp_llm_engine/core/status.h"
#include "cpp_llm_engine/cuda/cuda_decoder_layer.h"
#include "cpp_llm_engine/cuda/cuda_tensor.h"

#include <cstddef>
#include <vector>

namespace cpp_llm_engine::cuda {

class CudaExecutionContext;
class CudaKvCache;
class CudaPagedKvCache;

class CudaDecoderBatchLayer final {
public:
    static core::Status create(
        const CudaDecoderLayerConfig& config,
        std::size_t batch_size,
        CudaDecoderBatchLayer* output);

    core::Status forward(
        CudaExecutionContext* context,
        const CudaDecoderLayerWeights& weights,
        const std::vector<CudaKvCache*>& caches,
        CudaTensor* hidden,
        const std::vector<std::size_t>& positions);
    core::Status prefill_sequence(
        CudaExecutionContext* context,
        const CudaDecoderLayerWeights& weights,
        CudaKvCache* cache,
        CudaTensor* hidden,
        std::size_t position_offset);
    core::Status prefill_sequence_paged(
        CudaExecutionContext* context,
        const CudaDecoderLayerWeights& weights,
        CudaPagedKvCache* cache,
        std::size_t layer,
        const std::vector<std::size_t>& physical_blocks,
        CudaTensor* hidden,
        std::size_t position_offset);
    core::Status forward_paged(
        CudaExecutionContext* context,
        const CudaDecoderLayerWeights& weights,
        CudaPagedKvCache* cache,
        std::size_t layer,
        const std::vector<std::vector<std::size_t>>& physical_blocks,
        CudaTensor* hidden,
        const std::vector<std::size_t>& positions);
    std::size_t device_bytes() const;

private:
    CudaDecoderLayerConfig config_;
    std::size_t batch_size_ = 0;
    CudaTensor hidden_backup_;
    CudaTensor attention_normalized_;
    CudaTensor query_;
    CudaTensor key_;
    CudaTensor value_;
    CudaTensor attention_context_;
    CudaTensor projection_;
    CudaTensor ffn_normalized_;
    CudaTensor gate_;
    CudaTensor up_;
    CudaTensor swiglu_;
    CudaTensor down_;
    std::vector<CudaTensor> query_rows_;
    std::vector<CudaTensor> key_rows_;
    std::vector<CudaTensor> value_rows_;
    std::vector<CudaTensor> attention_rows_;
};

}  // namespace cpp_llm_engine::cuda
