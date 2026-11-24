// CUDA Backend：声明固定长度视觉序列的批量双向注意力。
// 职责：合并 Q/K/V 计算，同时用 batch 边界隔离不同图片的 token。
#pragma once

#include "cpp_llm_engine/cuda/cuda_status.h"

#include <cstddef>

namespace cpp_llm_engine::cuda {

class CudaStream;

struct CudaVisionAttentionRequest final {
    const float* query = nullptr;
    const float* key = nullptr;
    const float* value = nullptr;
    float* output = nullptr;
    std::size_t batch_size = 0;
    std::size_t tokens_per_image = 0;
    std::size_t head_count = 0;
    std::size_t head_dimension = 0;
    float scale = 0.0F;

    std::size_t score_count() const;
};

CudaStatus launch_vision_attention(
    const CudaVisionAttentionRequest& request,
    float* score_workspace,
    std::size_t workspace_elements,
    const CudaStream& stream);

}  // namespace cpp_llm_engine::cuda
