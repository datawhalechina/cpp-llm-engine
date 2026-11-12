// CUDA Backend：声明直接读取 packed 权重的融合 GEMV。
// 职责：在一个 kernel 中完成单行或批量输入的反量化、点积和行归约。
// 限制：批量路径最多让 4 个输入行复用一次权重解码，面向小批量 Decode。
#pragma once

#include "cpp_llm_engine/core/backend.h"
#include "cpp_llm_engine/cuda/cuda_status.h"

#include <cstddef>
#include <string_view>

namespace cpp_llm_engine::cuda {

class CudaQuantizedMatrix;
class CudaStream;

enum class CudaQuantizedGemvPath {
    FixedTile4,
    Adaptive,
};

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
    CudaStatus compute_batch(
        const CudaQuantizedMatrix& weights,
        const float* input,
        std::size_t batch_size,
        std::size_t input_width,
        float* output,
        std::size_t output_width,
        const CudaStream& stream,
        CudaQuantizedGemvPath path = CudaQuantizedGemvPath::Adaptive) const;
};

const CudaQuantizedGemvBackend& cuda_quantized_gemv_backend();

}  // namespace cpp_llm_engine::cuda
