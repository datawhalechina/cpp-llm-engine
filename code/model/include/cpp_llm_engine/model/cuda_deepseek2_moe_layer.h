// DeepSeek2 CUDA Model：声明请求级 MLA 与 MoE Decoder 层。
#pragma once

#include "cpp_llm_engine/model/cuda_deepseek2_dense_layer.h"
#include "cpp_llm_engine/model/cuda_deepseek2_moe_feed_forward.h"

namespace cpp_llm_engine::model {

struct CudaDeepSeek2MoeLayerConfig final {
    CudaDeepSeek2DenseLayerConfig attention;
    CudaDeepSeek2MoeConfig moe;

    core::Status validate() const;
};

class CudaDeepSeek2MoeLayer final {
public:
    static core::Status create(
        const CudaDeepSeek2MoeLayerConfig& config,
        std::size_t capacity_tokens,
        CudaDeepSeek2MoeLayer* output);

    core::Status forward(
        cuda::CudaExecutionContext* context,
        const CudaDeepSeek2MoeWeights& weights,
        cuda::CudaTensor* hidden,
        std::size_t position,
        CudaDeepSeek2MoeRoute* route);
    core::Status forward_attention(
        cuda::CudaExecutionContext* context,
        const CudaDeepSeek2MoeWeights& weights,
        cuda::CudaTensor* hidden,
        std::size_t position);
    core::Status finish_moe(
        cuda::CudaExecutionContext* context,
        cuda::CudaTensor* hidden);
    const cuda::CudaTensor& normalized_ffn_input() const;
    cuda::CudaTensor* moe_output();
    void reset();
    void trim(std::size_t position);
    std::size_t position() const;
    const cuda::CudaMlaCache& cache() const;

private:
    CudaDeepSeek2MoeLayerConfig config_;
    cuda::CudaMlaCache cache_;
    CudaDeepSeek2MoeFeedForward moe_;
    cuda::CudaTensor attention_norm_;
    cuda::CudaTensor query_;
    cuda::CudaTensor compressed_kv_;
    cuda::CudaTensor query_nope_;
    cuda::CudaTensor query_rope_;
    cuda::CudaTensor latent_;
    cuda::CudaTensor normalized_latent_;
    cuda::CudaTensor key_rope_;
    cuda::CudaTensor attention_probabilities_;
    cuda::CudaTensor attention_value_;
    cuda::CudaTensor attention_output_;
    cuda::CudaTensor attention_residual_;
    cuda::CudaTensor ffn_norm_;
    cuda::CudaTensor ffn_output_;
};

}  // namespace cpp_llm_engine::model
