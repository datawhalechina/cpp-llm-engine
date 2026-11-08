// CUDA Backend：声明直接读取 Q4_K/Q6_K packed 权重的融合 GEMV。
// 职责：在一个 kernel 中完成反量化、点积和行归约。
// 限制：当前只处理单输入向量；Tensor Batch 仍使用反量化加 SGEMM。
#pragma once

#include "cpp_llm_engine/core/backend.h"
#include "cpp_llm_engine/cuda/cuda_status.h"

#include <cstddef>
#include <string_view>

namespace cpp_llm_engine::cuda {

class CudaQuantizedMatrix;
class CudaStream;

class CudaQuantizedGemvBackend final : public core::Backend {
public:
    core::BackendKind kind() const override;
    std::string_view name() const override;
    CudaStatus compute(
        const CudaQuantizedMatrix& weights,
        const float* input,
        std::size_t input_elements,
        float* output,
        std::size_t output_elements,
        const CudaStream& stream) const;
};

const CudaQuantizedGemvBackend& cuda_quantized_gemv_backend();

}  // namespace cpp_llm_engine::cuda
