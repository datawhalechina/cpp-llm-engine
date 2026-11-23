// CUDA Backend：声明视觉 Block 使用的 LayerNorm、bias 广播与 GELU。
// 职责：补齐 SigLIP 与语言 Decoder 不同的逐元素语义。
// 验证：多模态 CUDA 单 Block 与 CPU reference 对照。
#pragma once

#include "cpp_llm_engine/cuda/cuda_status.h"

#include <cstddef>

namespace cpp_llm_engine::cuda {

class CudaStream;

struct CudaLayerNormBatchRequest final {
    const float* input = nullptr;
    const float* weight = nullptr;
    const float* bias = nullptr;
    float* output = nullptr;
    std::size_t rows = 0;
    std::size_t width = 0;
    float epsilon = 0.0F;
};

struct CudaBiasBatchRequest final {
    const float* input = nullptr;
    const float* bias = nullptr;
    float* output = nullptr;
    std::size_t elements = 0;
    std::size_t width = 0;
};

struct CudaGeluRequest final {
    const float* input = nullptr;
    float* output = nullptr;
    std::size_t elements = 0;
};

CudaStatus launch_layer_norm_batch(
    const CudaLayerNormBatchRequest& request, const CudaStream& stream);
CudaStatus launch_bias_batch(
    const CudaBiasBatchRequest& request, const CudaStream& stream);
CudaStatus launch_gelu(
    const CudaGeluRequest& request, const CudaStream& stream);

}  // namespace cpp_llm_engine::cuda
