// CUDA Decoder：声明可验证的单 token Llama Decoder 层组装。
// 职责：按 RMSNorm、QKV、RoPE、GQA、Residual、SwiGLU 顺序调用既有 backend。
// 契约：权重由 Model 只读持有，KV Cache 由请求持有，层对象只拥有中间 Tensor。
// 验证：合成小维度 CPU 逐阶段 golden。
#pragma once

#include "cpp_llm_engine/core/status.h"
#include "cpp_llm_engine/cuda/cuda_tensor.h"
#include "cpp_llm_engine/cuda/cuda_linear_weight.h"

#include <cstddef>
#include <vector>

namespace cpp_llm_engine::cuda {

class CudaExecutionContext;
class CudaKvCache;
class CudaPagedKvCache;

struct CudaDecoderLayerConfig final {
    std::size_t hidden_size = 0;
    std::size_t ffn_size = 0;
    std::size_t query_heads = 0;
    std::size_t kv_heads = 0;
    std::size_t head_dimension = 0;
    std::size_t rotary_dimension = 0;
    float rms_epsilon = 1.0e-5F;
    float rope_frequency_base = 10000.0F;
};

struct CudaDecoderLayerWeights final {
    const CudaTensor* attention_norm = nullptr;
    CudaLinearWeight query;
    CudaLinearWeight key;
    CudaLinearWeight value;
    CudaLinearWeight attention_output;
    const CudaTensor* ffn_norm = nullptr;
    CudaLinearWeight gate;
    CudaLinearWeight up;
    CudaLinearWeight down;
};

class CudaDecoderLayer final {
public:
    static core::Status create(
        const CudaDecoderLayerConfig& config, CudaDecoderLayer* output);
    core::Status forward(
        CudaExecutionContext* context,
        const CudaDecoderLayerWeights& weights,
        CudaKvCache* cache,
        CudaTensor* hidden,
        std::size_t position);
    core::Status forward_paged(
        CudaExecutionContext* context,
        const CudaDecoderLayerWeights& weights,
        CudaPagedKvCache* cache,
        std::size_t layer,
        const std::vector<std::size_t>& physical_blocks,
        CudaTensor* hidden,
        std::size_t position);
    const CudaTensor& attention_normalized() const;
    const CudaTensor& attention_context() const;
    const CudaTensor& ffn_normalized() const;
    const CudaTensor& swiglu_output() const;

private:
    CudaDecoderLayerConfig config_;
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
};

}  // namespace cpp_llm_engine::cuda
