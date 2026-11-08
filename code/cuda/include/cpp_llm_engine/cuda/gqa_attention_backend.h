// CUDA Backend：声明三阶段 causal GQA 与融合在线 Softmax 实验入口。
// 职责：保留外部 score workspace golden，并提供不物化完整 score 的候选路径。
// 验证：prefill/decode CPU golden、GQA head 映射、错误参数和 A/B event 计时。
#pragma once

#include "cpp_llm_engine/core/backend.h"
#include "cpp_llm_engine/core/gqa_attention_contract.h"
#include "cpp_llm_engine/cuda/cuda_status.h"

#include <cstddef>
#include <string_view>

namespace cpp_llm_engine::cuda {

class CudaStream;

struct CudaPagedGqaAttentionRequest final {
    const float* query = nullptr;
    const float* key = nullptr;
    const float* value = nullptr;
    const std::size_t* physical_blocks = nullptr;
    float* output = nullptr;
    std::size_t query_tokens = 0;
    std::size_t key_tokens = 0;
    std::size_t query_head_count = 0;
    std::size_t kv_head_count = 0;
    std::size_t head_dimension = 0;
    std::size_t query_position_offset = 0;
    std::size_t layer = 0;
    std::size_t layer_count = 0;
    std::size_t total_blocks = 0;
    std::size_t block_size = 0;
    std::size_t physical_block_count = 0;
    bool causal = true;
    float scale = 0.0F;

    core::Status validate() const;
    std::size_t score_count() const;
};

class CudaGqaAttentionBackend final : public core::Backend {
public:
    core::BackendKind kind() const override;
    std::string_view name() const override;

    CudaStatus compute(
        const core::GqaAttentionRequest& request,
        float* score_workspace,
        std::size_t workspace_elements,
        const CudaStream& stream) const;
    CudaStatus compute_online(
        const core::GqaAttentionRequest& request,
        const CudaStream& stream) const;
    CudaStatus compute_paged(
        const CudaPagedGqaAttentionRequest& request,
        float* score_workspace,
        std::size_t workspace_elements,
        const CudaStream& stream) const;
};

const CudaGqaAttentionBackend& cuda_gqa_attention_backend();

}  // namespace cpp_llm_engine::cuda
