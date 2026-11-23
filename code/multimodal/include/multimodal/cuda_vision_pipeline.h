// 多模态 CUDA：SmolVLM 视觉设备流水线
// 职责：串联真实 Patch/Position、SigLIP Encoder 和 IDEFICS3 Projector。
// 契约：图片解码与预处理留在 Host；视觉 hidden 在三个设备阶段之间不回传。
// 验证：multimodal_cuda_vision_pipeline_inspect 与 CPU reference 对照。
#pragma once

#include "cpp_llm_engine/core/status.h"
#include "cpp_llm_engine/cuda/cuda_tensor.h"
#include "multimodal/cuda_vision_embedding.h"
#include "multimodal/cuda_vision_encoder.h"
#include "multimodal/cuda_vision_projector.h"
#include "multimodal/mmproj_schema.h"

#include <cstddef>
#include <string>
#include <vector>

namespace cpp_llm_engine::cuda {
class CudaExecutionContext;
}

namespace multimodal {

class CudaSmolvlmVisionPipeline final {
public:
    static cpp_llm_engine::core::Status upload_gguf(
        cpp_llm_engine::cuda::CudaExecutionContext* context,
        const std::string& path,
        const chapter01::GgufModelIndex& index,
        const VisionModelConfig& config,
        const VisionTensorLayout& layout,
        CudaSmolvlmVisionPipeline* output);

    cpp_llm_engine::core::Status forward(
        cpp_llm_engine::cuda::CudaExecutionContext* context,
        const PatchTensor& patches,
        const std::vector<std::size_t>& positions,
        std::size_t grid_width,
        std::size_t grid_height,
        cpp_llm_engine::cuda::CudaTensor* output);

    bool valid() const;
    std::size_t workspace_rebuild_count() const;

private:
    cpp_llm_engine::core::Status prepare_workspace(std::size_t token_count);

    CudaPatchEmbedding embedding_;
    CudaSiglipEncoder encoder_;
    CudaIdefics3Projector projector_;
    cpp_llm_engine::cuda::CudaTensor embedded_;
    cpp_llm_engine::cuda::CudaTensor encoded_;
    std::size_t merge_size_ = 0;
    std::size_t projection_size_ = 0;
    std::size_t prepared_tokens_ = 0;
    std::size_t workspace_rebuild_count_ = 0;
};

}  // namespace multimodal
