// CUDA Backend：声明原地 Rotary Position Embedding backend。
// 职责：把 Core RoPE 连续布局映射到独立 pair 线程并排入显式 stream。
// 验证：多 token/head CPU golden、局部 rotary dimension、错误参数和 event 计时。
#pragma once

#include "cpp_llm_engine/core/backend.h"
#include "cpp_llm_engine/core/rope_contract.h"
#include "cpp_llm_engine/cuda/cuda_status.h"

#include <string_view>

namespace cpp_llm_engine::cuda {

class CudaStream;

class CudaRopeBackend final : public core::Backend {
public:
    core::BackendKind kind() const override;
    std::string_view name() const override;

    CudaStatus apply(
        const core::RotaryEmbeddingRequest& request,
        const CudaStream& stream) const;
    CudaStatus apply_batch(
        const core::RotaryEmbeddingBatchRequest& request,
        const CudaStream& stream) const;
};

const CudaRopeBackend& cuda_rope_backend();

}  // namespace cpp_llm_engine::cuda
