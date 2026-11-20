// DeepSeek2 CUDA Model：声明请求级稠密 Decoder 层与压缩 MLA Context。
// 职责：编排量化 Linear、YaRN、压缩 Cache、MLA、SwiGLU 和双残差。
// 契约：forward 成功提交一个缓存词元；失败回退到调用前位置。
#pragma once

#include "cpp_llm_engine/core/rope_contract.h"
#include "cpp_llm_engine/core/status.h"
#include "cpp_llm_engine/cuda/cuda_mla_cache.h"
#include "cpp_llm_engine/cuda/cuda_paged_mla_cache.h"
#include "cpp_llm_engine/cuda/cuda_tensor.h"

#include <cstddef>
#include <vector>

namespace cpp_llm_engine::cuda { class CudaExecutionContext; }

namespace cpp_llm_engine::model {

struct CudaDeepSeek2DenseLayerWeightView;
class CudaDeepSeek2DensePrefillWorkspace;
class CudaDeepSeek2MoeLayer;

struct CudaDeepSeek2PagedMlaView final {
    cuda::CudaPagedMlaCache* cache = nullptr;
    const std::vector<std::size_t>* physical_blocks = nullptr;
    std::size_t layer = 0;
    std::size_t physical_block = 0;
    std::size_t slot = 0;
    std::size_t token_count = 0;

    bool valid() const;
};

struct CudaDeepSeek2PagedMlaBatchView final {
    cuda::CudaPagedMlaCache* cache = nullptr;
    const std::vector<std::size_t>* physical_blocks = nullptr;
    std::size_t layer = 0;
    std::size_t token_count = 0;

    bool valid() const;
};

struct CudaDeepSeek2DenseLayerConfig final {
    std::size_t hidden_size = 0;
    std::size_t head_count = 0;
    std::size_t nope_dimension = 0;
    std::size_t rope_dimension = 0;
    std::size_t value_dimension = 0;
    std::size_t latent_dimension = 0;
    std::size_t ffn_dimension = 0;
    float rms_norm_epsilon = 1.0e-6F;
    float rope_frequency_base = 10000.0F;
    core::YarnRotaryScaling yarn;
    float attention_scale = 0.0F;

    core::Status validate() const;
};

class CudaDeepSeek2DenseLayer final {
public:
    static core::Status create(
        const CudaDeepSeek2DenseLayerConfig& config,
        std::size_t capacity_tokens,
        CudaDeepSeek2DenseLayer* output);

    core::Status forward(
        cuda::CudaExecutionContext* context,
        const CudaDeepSeek2DenseLayerWeightView& weights,
        cuda::CudaTensor* hidden,
        std::size_t position,
        const CudaDeepSeek2PagedMlaView* paged = nullptr);
    core::Status prefill_sequence(
        cuda::CudaExecutionContext* context,
        const CudaDeepSeek2DenseLayerWeightView& weights,
        cuda::CudaTensor* hidden,
        std::size_t position_offset,
        CudaDeepSeek2DensePrefillWorkspace* workspace,
        const CudaDeepSeek2PagedMlaBatchView* paged = nullptr);
    void reset();
    void trim(std::size_t position);
    std::size_t position() const;
    const cuda::CudaMlaCache& cache() const;

private:
    CudaDeepSeek2DenseLayerConfig config_;
    cuda::CudaMlaCache cache_;
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
    cuda::CudaTensor gate_;
    cuda::CudaTensor up_;
    cuda::CudaTensor activated_;
    cuda::CudaTensor ffn_output_;
};

class CudaDeepSeek2DensePrefillWorkspace final {
public:
    static core::Status create(
        const CudaDeepSeek2DenseLayerConfig& config,
        std::size_t token_count,
        std::size_t capacity_tokens,
        CudaDeepSeek2DensePrefillWorkspace* output);

    std::size_t token_count() const;

private:
    friend class CudaDeepSeek2DenseLayer;
    friend class CudaDeepSeek2MoeLayer;
    std::size_t token_count_ = 0;
    cuda::CudaTensor hidden_backup_;
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
    cuda::CudaTensor gate_;
    cuda::CudaTensor up_;
    cuda::CudaTensor activated_;
    cuda::CudaTensor ffn_output_;
    std::vector<cuda::CudaTensor> moe_input_rows_;
    std::vector<cuda::CudaTensor> moe_output_rows_;
};

}  // namespace cpp_llm_engine::model
