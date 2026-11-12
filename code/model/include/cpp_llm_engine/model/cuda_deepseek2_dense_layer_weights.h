// DeepSeek2 CUDA Model：声明稠密层 host 权重到 device 权重的上传边界。
// 职责：保存 MLA、稠密 FFN 的归一化向量和 packed Linear，并提供只读视图。
// 契约：上传成功后不依赖 Chapter01 Tensor；失败不覆盖已有 device 权重。
#pragma once

#include "cpp_llm_engine/core/status.h"
#include "cpp_llm_engine/cuda/cuda_linear_weight.h"
#include "cpp_llm_engine/cuda/cuda_quantized_matrix.h"
#include "cpp_llm_engine/cuda/cuda_tensor.h"

namespace chapter01 { struct RealDeepSeek2DenseLayerWeights; }

namespace cpp_llm_engine::cuda { class CudaExecutionContext; }

namespace cpp_llm_engine::model {

struct CudaDeepSeek2DenseLayerWeightView final {
    const cuda::CudaTensor* attention_norm = nullptr;
    cuda::CudaLinearWeight query;
    cuda::CudaLinearWeight kv_a_mqa;
    const cuda::CudaTensor* kv_a_norm = nullptr;
    cuda::CudaLinearWeight kv_b;
    const cuda::CudaTensor* kv_b_dequantized = nullptr;
    cuda::CudaLinearWeight attention_output;
    const cuda::CudaTensor* ffn_norm = nullptr;
    cuda::CudaLinearWeight ffn_gate;
    cuda::CudaLinearWeight ffn_up;
    cuda::CudaLinearWeight ffn_down;

    bool valid() const;
};

class CudaDeepSeek2DenseLayerWeights final {
public:
    static core::Status upload(
        cuda::CudaExecutionContext* context,
        const chapter01::RealDeepSeek2DenseLayerWeights& host,
        CudaDeepSeek2DenseLayerWeights* output);

    CudaDeepSeek2DenseLayerWeightView view() const;
    bool valid() const;

private:
    cuda::CudaTensor attention_norm_;
    cuda::CudaQuantizedMatrix query_;
    cuda::CudaQuantizedMatrix kv_a_mqa_;
    cuda::CudaTensor kv_a_norm_;
    cuda::CudaQuantizedMatrix kv_b_;
    cuda::CudaTensor kv_b_dequantized_;
    cuda::CudaQuantizedMatrix attention_output_;
    cuda::CudaTensor ffn_norm_;
    cuda::CudaQuantizedMatrix ffn_gate_;
    cuda::CudaQuantizedMatrix ffn_up_;
    cuda::CudaQuantizedMatrix ffn_down_;
};

}  // namespace cpp_llm_engine::model
