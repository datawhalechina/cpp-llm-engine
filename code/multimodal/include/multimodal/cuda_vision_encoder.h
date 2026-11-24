// 多模态 CUDA：SigLIP Vision Encoder 设备接口
// 职责：串联常驻 Block 权重，复用 workspace 和层间 hidden buffer。
// 契约：输入/输出为 [tokens,hidden] Float32 CUDA Tensor。
#pragma once

#include "multimodal/cuda_siglip_block.h"

#include <cstddef>
#include <string>
#include <vector>

namespace multimodal {

class CudaSiglipEncoder final {
public:
    static cpp_llm_engine::core::Status upload(
        cpp_llm_engine::cuda::CudaExecutionContext* context,
        const std::vector<SiglipBlockWeights>& blocks,
        const LayerNormWeights& post_layer_norm,
        std::size_t attention_heads,
        float layer_norm_epsilon,
        CudaSiglipEncoder* output);

    static cpp_llm_engine::core::Status upload_gguf(
        cpp_llm_engine::cuda::CudaExecutionContext* context,
        const std::string& path,
        const chapter01::GgufModelIndex& index,
        const VisionModelConfig& config,
        const VisionTensorLayout& layout,
        CudaSiglipEncoder* output);

    cpp_llm_engine::core::Status forward(
        cpp_llm_engine::cuda::CudaExecutionContext* context,
        const cpp_llm_engine::cuda::CudaTensor& input,
        cpp_llm_engine::cuda::CudaTensor* output);
    cpp_llm_engine::core::Status forward_batch(
        cpp_llm_engine::cuda::CudaExecutionContext* context,
        const cpp_llm_engine::cuda::CudaTensor& input,
        std::size_t batch_size,
        cpp_llm_engine::cuda::CudaTensor* output);

    bool valid() const;
    std::size_t block_count() const;
    std::size_t workspace_rebuild_count() const;

private:
    cpp_llm_engine::core::Status prepare_workspace(
        cpp_llm_engine::cuda::CudaExecutionContext* context,
        std::size_t token_count);

    std::vector<CudaSiglipBlock> blocks_;
    cpp_llm_engine::cuda::CudaTensor post_norm_weight_;
    cpp_llm_engine::cuda::CudaTensor post_norm_bias_;
    CudaSiglipBlockWorkspace workspace_;
    cpp_llm_engine::cuda::CudaTensor hidden_a_;
    cpp_llm_engine::cuda::CudaTensor hidden_b_;
    std::size_t hidden_size_ = 0;
    std::size_t attention_heads_ = 0;
    std::size_t feed_forward_size_ = 0;
    std::size_t prepared_tokens_ = 0;
    std::size_t workspace_rebuild_count_ = 0;
    float layer_norm_epsilon_ = 0.0F;
};

}  // namespace multimodal
