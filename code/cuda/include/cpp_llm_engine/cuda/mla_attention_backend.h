// CUDA Backend：声明压缩 Multi-head Latent Attention 数据面。
// 职责：直接读取 latent/rotary Cache 和展开矩阵，避免物化历史完整 K/V。
// 验证：双词元 CPU golden、缓存增长、容量失败和请求隔离。
#pragma once

#include "cpp_llm_engine/core/backend.h"
#include "cpp_llm_engine/cuda/cuda_status.h"

#include <cstddef>
#include <string_view>

namespace cpp_llm_engine::cuda {

class CudaMlaCache;
class CudaStream;
class CudaTensor;

struct CudaMlaPrepareRequest final {
    const float* query = nullptr;
    const float* compressed_kv = nullptr;
    float* query_nope = nullptr;
    float* query_rope = nullptr;
    float* latent = nullptr;
    float* key_rope = nullptr;
    std::size_t head_count = 0;
    std::size_t nope_dimension = 0;
    std::size_t rope_dimension = 0;
    std::size_t latent_dimension = 0;

    core::Status validate() const;
};

struct CudaMlaAttentionRequest final {
    const float* query_nope = nullptr;
    const float* query_rope = nullptr;
    const float* kv_b = nullptr;
    float* probabilities = nullptr;
    float* output = nullptr;
    std::size_t head_count = 0;
    std::size_t nope_dimension = 0;
    std::size_t rope_dimension = 0;
    std::size_t value_dimension = 0;
    std::size_t latent_dimension = 0;
    std::size_t probability_capacity_tokens = 0;
    float scale = 0.0F;

    core::Status validate() const;
};

class CudaMlaAttentionBackend final : public core::Backend {
public:
    core::BackendKind kind() const override;
    std::string_view name() const override;

    CudaStatus compute(
        const CudaMlaAttentionRequest& request,
        const CudaMlaCache& cache,
        const CudaStream& stream) const;
    CudaStatus prepare(
        const CudaMlaPrepareRequest& request,
        const CudaStream& stream) const;
};

const CudaMlaAttentionBackend& cuda_mla_attention_backend();

}  // namespace cpp_llm_engine::cuda
