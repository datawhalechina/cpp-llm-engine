// 多模态 CUDA：Patch/Position Embedding 设备适配接口
// 职责：上传 F32 投影权重并复用 CUDA Linear、行收集和向量加法。
// 契约：Patch 权重从 output-major 转换为 CUDA [input,output] 布局。
// 验证：multimodal_cuda_vision_embedding_test 与 CPU reference 对照。
#pragma once

#include "cpp_llm_engine/core/status.h"
#include "cpp_llm_engine/cuda/cuda_tensor.h"
#include "multimodal/vision_encoder.h"

#include <cstddef>
#include <vector>

namespace cpp_llm_engine::cuda {
class CudaExecutionContext;
}

namespace multimodal {

class CudaPatchEmbedding final {
public:
    static cpp_llm_engine::core::Status upload(
        cpp_llm_engine::cuda::CudaExecutionContext* context,
        const PatchEmbeddingWeights& host,
        CudaPatchEmbedding* output);

    cpp_llm_engine::core::Status forward(
        cpp_llm_engine::cuda::CudaExecutionContext* context,
        const PatchTensor& patches,
        const std::vector<std::size_t>& positions,
        cpp_llm_engine::cuda::CudaTensor* output) const;

    bool valid() const;
    std::size_t patch_values() const;
    std::size_t hidden_size() const;
    std::size_t position_count() const;

private:
    cpp_llm_engine::cuda::CudaTensor patch_weight_;
    cpp_llm_engine::cuda::CudaTensor position_and_bias_;
    std::size_t patch_values_ = 0;
    std::size_t hidden_size_ = 0;
    std::size_t position_count_ = 0;
};

}  // namespace multimodal
