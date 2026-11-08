// CUDA Backend：声明 RMSNorm 与 SwiGLU backend。
// 职责：补齐单层 Llama Decoder 在 Linear/Attention 之外的逐元素算子。
// 验证：多宽度 CPU golden、非法参数与非整除 grid 测试。
#pragma once

#include "cpp_llm_engine/core/backend.h"
#include "cpp_llm_engine/core/decoder_elementwise_contract.h"
#include "cpp_llm_engine/cuda/cuda_status.h"

#include <string_view>

namespace cpp_llm_engine::cuda {

class CudaStream;

class CudaDecoderElementwiseBackend final : public core::Backend {
public:
    core::BackendKind kind() const override;
    std::string_view name() const override;
    CudaStatus rms_norm(
        const core::RmsNormRequest& request, const CudaStream& stream) const;
    CudaStatus rms_norm_batch(
        const core::RmsNormBatchRequest& request,
        const CudaStream& stream) const;
    CudaStatus swiglu(
        const core::SwiGluRequest& request, const CudaStream& stream) const;
};

const CudaDecoderElementwiseBackend& cuda_decoder_elementwise_backend();

}  // namespace cpp_llm_engine::cuda
