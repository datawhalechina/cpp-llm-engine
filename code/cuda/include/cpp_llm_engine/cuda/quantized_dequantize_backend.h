// CUDA Backend：声明 packed 量化 payload 的 Float32 反量化 kernel。
// 职责：只负责 device-to-device 解码；Linear 的 SGEMM 由 ExecutionContext 编排。
// 验证：与 Chapter05 CPU reference 逐元素和最终 Linear 输出对照。
#pragma once

#include "cpp_llm_engine/core/backend.h"
#include "cpp_llm_engine/cuda/cuda_status.h"

#include <cstddef>
#include <string_view>

namespace cpp_llm_engine::cuda {

class CudaQuantizedMatrix;
class CudaStream;

class CudaQuantizedDequantizeBackend final : public core::Backend {
public:
    core::BackendKind kind() const override;
    std::string_view name() const override;
    CudaStatus dequantize(
        const CudaQuantizedMatrix& weights,
        float* output,
        std::size_t output_elements,
        const CudaStream& stream) const;
    CudaStatus dequantize_row(
        const CudaQuantizedMatrix& weights,
        std::size_t row,
        float* output,
        std::size_t output_elements,
        const CudaStream& stream) const;
};

const CudaQuantizedDequantizeBackend& cuda_quantized_dequantize_backend();

}  // namespace cpp_llm_engine::cuda
