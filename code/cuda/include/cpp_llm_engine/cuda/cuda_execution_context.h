// CUDA 执行层：声明统一拥有 stream、cuBLAS handle 和 workspace 的上下文。
// 职责：分派算子、KV append 与 GQA，不向上层暴露裸分配或 stream。
// 验证：真实 GPU CTest 覆盖复制、workspace、RoPE、KV 和 GQA 错误路径。
#pragma once

#include "cpp_llm_engine/core/rope_contract.h"
#include "cpp_llm_engine/core/status.h"
#include "cpp_llm_engine/cuda/cublas_matmul_backend.h"
#include "cpp_llm_engine/cuda/cuda_quantized_matrix.h"
#include "cpp_llm_engine/cuda/cuda_linear_weight.h"
#include "cpp_llm_engine/cuda/cuda_stream.h"
#include "cpp_llm_engine/cuda/cuda_tensor.h"
#include "cpp_llm_engine/cuda/cuda_workspace.h"
#include "cpp_llm_engine/cuda/quantized_gemv_backend.h"

#include <cstddef>
#include <vector>

namespace cpp_llm_engine::cuda {

enum class CudaAttentionPath {
    ThreeStage,
    FusedOnline,
};

class CudaKvCache;
class CudaMlaCache;
class CudaPagedKvCache;

class CudaExecutionContext final {
public:
    CudaExecutionContext() = default;
    ~CudaExecutionContext() = default;

    CudaExecutionContext(const CudaExecutionContext&) = delete;
    CudaExecutionContext& operator=(const CudaExecutionContext&) = delete;
    CudaExecutionContext(CudaExecutionContext&&) noexcept = default;
    CudaExecutionContext& operator=(CudaExecutionContext&&) noexcept = default;

    static core::Status create(CudaExecutionContext* output);

    core::Status upload(CudaTensor* tensor, const void* source, std::size_t bytes);
    core::Status upload(
        CudaQuantizedMatrix* matrix,
        const void* source,
        std::size_t bytes);
    core::Status download(
        const CudaTensor& tensor,
        void* destination,
        std::size_t bytes) const;
    core::Status copy(
        const CudaTensor& source,
        CudaTensor* destination) const;
    core::Status pack_rows(
        const std::vector<const CudaTensor*>& rows,
        CudaTensor* batch) const;
    core::Status unpack_rows(
        const CudaTensor& batch,
        const std::vector<CudaTensor*>& rows) const;
    core::Status vector_add(
        const CudaTensor& lhs,
        const CudaTensor& rhs,
        CudaTensor* output) const;
    core::Status rms_norm(
        const CudaTensor& input,
        const CudaTensor& weight,
        CudaTensor* output,
        float epsilon) const;
    core::Status rms_norm_batch(
        const CudaTensor& input,
        const CudaTensor& weight,
        CudaTensor* output,
        float epsilon) const;
    core::Status swiglu(
        const CudaTensor& gate,
        const CudaTensor& up,
        CudaTensor* output) const;
    core::Status matmul(
        const CudaTensor& lhs,
        const CudaTensor& rhs,
        CudaTensor* output) const;
    core::Status linear(
        const CudaTensor& input,
        const CudaTensor& weights,
        CudaTensor* output) const;
    core::Status linear(
        const CudaTensor& input,
        const CudaLinearWeight& weights,
        CudaTensor* output);
    core::Status linear_batch(
        const CudaTensor& input,
        const CudaLinearWeight& weights,
        CudaTensor* output);
    core::Status quantized_linear(
        const CudaQuantizedMatrix& weights,
        const CudaTensor& input,
        CudaTensor* output);
    core::Status quantized_linear_batch(
        const CudaQuantizedMatrix& weights,
        const CudaTensor& input,
        CudaTensor* output);
    core::Status embedding_lookup(
        const CudaQuantizedMatrix& table,
        std::size_t row,
        CudaTensor* output) const;
    core::Status dequantize(
        const CudaQuantizedMatrix& matrix,
        CudaTensor* output) const;
    core::Status prepare_mla(
        const CudaTensor& query,
        const CudaTensor& compressed_kv,
        CudaTensor* query_nope,
        CudaTensor* query_rope,
        CudaTensor* latent,
        CudaTensor* key_rope) const;
    core::Status mla_attention(
        const CudaTensor& query_nope,
        const CudaTensor& query_rope,
        const CudaTensor& kv_b,
        const CudaMlaCache& cache,
        CudaTensor* probabilities,
        CudaTensor* output,
        float scale) const;
    core::Status append_mla(
        CudaMlaCache* cache,
        const CudaTensor& latent,
        const CudaTensor& key_rope) const;
    core::Status rope(
        CudaTensor* values,
        std::size_t position_offset,
        std::size_t rotary_dimension,
        float frequency_base,
        core::RotaryPairing pairing = core::RotaryPairing::Adjacent,
        const core::YarnRotaryScaling& yarn = {}) const;
    core::Status rope_batch(
        CudaTensor* values,
        const std::vector<std::size_t>& positions,
        std::size_t rotary_dimension,
        float frequency_base,
        core::RotaryPairing pairing = core::RotaryPairing::Adjacent,
        const core::YarnRotaryScaling& yarn = {});
    core::Status append_kv(
        CudaKvCache* cache,
        const CudaTensor& key,
        const CudaTensor& value) const;
    core::Status write_paged_kv(
        CudaPagedKvCache* cache,
        std::size_t layer,
        std::size_t physical_block,
        std::size_t slot,
        const CudaTensor& key,
        const CudaTensor& value) const;
    core::Status download_kv(
        const CudaKvCache& cache,
        float* key,
        float* value,
        std::size_t token_count) const;
    core::Status gqa_attention(
        const CudaTensor& query,
        const CudaKvCache& cache,
        CudaTensor* output,
        std::size_t query_position_offset,
        bool causal = true,
        float scale = 0.0F);
    core::Status paged_gqa_attention(
        const CudaTensor& query,
        const CudaPagedKvCache& cache,
        std::size_t layer,
        const std::vector<std::size_t>& physical_blocks,
        std::size_t key_tokens,
        CudaTensor* output,
        std::size_t query_position_offset,
        bool causal = true,
        float scale = 0.0F);
    core::Status reserve_workspace(std::size_t bytes);
    core::Status synchronize() const;

    void set_attention_path(CudaAttentionPath path);
    CudaAttentionPath attention_path() const;
    void set_quantized_gemv_path(CudaQuantizedGemvPath path);
    CudaQuantizedGemvPath quantized_gemv_path() const;

    std::size_t workspace_capacity_bytes() const;
    std::size_t workspace_allocation_count() const;
    bool valid() const;

private:
    CudaStream stream_;
    CublasMatmulBackend matmul_backend_;
    CudaWorkspace workspace_;
    CudaAttentionPath attention_path_ = CudaAttentionPath::ThreeStage;
    CudaQuantizedGemvPath quantized_gemv_path_ =
        CudaQuantizedGemvPath::Adaptive;
};

}  // namespace cpp_llm_engine::cuda
