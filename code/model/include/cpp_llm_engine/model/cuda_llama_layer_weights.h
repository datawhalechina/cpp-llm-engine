// Llama CUDA Model：声明 Chapter01 host layer 到 device weight 的上传适配器。
// 职责：保存两个 F32 norm 与七个 packed Linear，并生成 Decoder 只读权重视图。
// 契约：Model 拥有该对象；上传完成后不再依赖 Chapter01 Tensor 生命周期。
// 验证：可选真实 GPU 测试覆盖格式、方向、payload bytes、移动和失败路径。
#pragma once

#include "cpp_llm_engine/core/status.h"
#include "cpp_llm_engine/cuda/cuda_decoder_layer.h"
#include "cpp_llm_engine/cuda/cuda_quantized_matrix.h"
#include "cpp_llm_engine/cuda/cuda_tensor.h"

namespace chapter01 { struct RealLlamaLayerWeights; }

namespace cpp_llm_engine::model {

class CudaLlamaLayerWeights final {
public:
    static core::Status upload(
        cuda::CudaExecutionContext* context,
        const chapter01::RealLlamaLayerWeights& host,
        CudaLlamaLayerWeights* output);

    cuda::CudaDecoderLayerWeights decoder_weights() const;
    bool valid() const;
    const cuda::CudaQuantizedMatrix& query() const;
    const cuda::CudaQuantizedMatrix& down() const;

private:
    cuda::CudaTensor attention_norm_;
    cuda::CudaQuantizedMatrix query_;
    cuda::CudaQuantizedMatrix key_;
    cuda::CudaQuantizedMatrix value_;
    cuda::CudaQuantizedMatrix attention_output_;
    cuda::CudaTensor ffn_norm_;
    cuda::CudaQuantizedMatrix gate_;
    cuda::CudaQuantizedMatrix up_;
    cuda::CudaQuantizedMatrix down_;
};

}  // namespace cpp_llm_engine::model
