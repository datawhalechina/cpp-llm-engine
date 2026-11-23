// 多模态 CUDA：单个 SigLIP Block 设备接口
// 职责：组装标准 LayerNorm、双向多头注意力、GELU MLP 与双残差。
// 契约：输入/输出为 [tokens,hidden] Float32 CUDA Tensor。
// 验证：multimodal_cuda_siglip_block_test 与 CPU reference 对照。
#pragma once

#include "cpp_llm_engine/core/status.h"
#include "cpp_llm_engine/cuda/cuda_kv_cache.h"
#include "cpp_llm_engine/cuda/cuda_quantized_matrix.h"
#include "cpp_llm_engine/cuda/cuda_tensor.h"
#include "multimodal/mmproj_schema.h"
#include "multimodal/vision_encoder.h"

#include <cstddef>
#include <string>

namespace cpp_llm_engine::cuda {
class CudaExecutionContext;
class CudaLinearWeight;
}

namespace multimodal {

class CudaSiglipBlockWorkspace final {
public:
    static cpp_llm_engine::core::Status create(
        std::size_t token_count,
        std::size_t hidden_size,
        std::size_t attention_heads,
        std::size_t feed_forward_size,
        CudaSiglipBlockWorkspace* output);

    bool valid_for(
        std::size_t token_count,
        std::size_t hidden_size,
        std::size_t attention_heads,
        std::size_t feed_forward_size) const;
    std::size_t token_count() const;

private:
    friend class CudaSiglipBlock;

    cpp_llm_engine::cuda::CudaTensor normalized_attention;
    cpp_llm_engine::cuda::CudaTensor query;
    cpp_llm_engine::cuda::CudaTensor key;
    cpp_llm_engine::cuda::CudaTensor value;
    cpp_llm_engine::cuda::CudaTensor attention_context;
    cpp_llm_engine::cuda::CudaTensor attention_projection;
    cpp_llm_engine::cuda::CudaTensor attention_residual;
    cpp_llm_engine::cuda::CudaTensor normalized_feed_forward;
    cpp_llm_engine::cuda::CudaTensor feed_forward_hidden;
    cpp_llm_engine::cuda::CudaTensor feed_forward_activated;
    cpp_llm_engine::cuda::CudaTensor feed_forward_output;
    cpp_llm_engine::cuda::CudaKvCache cache;
    std::size_t token_count_ = 0;
    std::size_t hidden_size_ = 0;
    std::size_t attention_heads_ = 0;
    std::size_t feed_forward_size_ = 0;
};

class CudaSiglipBlock final {
public:
    static cpp_llm_engine::core::Status upload(
        cpp_llm_engine::cuda::CudaExecutionContext* context,
        const SiglipBlockWeights& host,
        std::size_t attention_heads,
        float layer_norm_epsilon,
        CudaSiglipBlock* output);

    static cpp_llm_engine::core::Status upload_gguf(
        cpp_llm_engine::cuda::CudaExecutionContext* context,
        const std::string& path,
        const chapter01::GgufModelIndex& index,
        const VisionModelConfig& config,
        const VisionTensorLayout& layout,
        std::size_t block_index,
        CudaSiglipBlock* output);

    cpp_llm_engine::core::Status forward(
        cpp_llm_engine::cuda::CudaExecutionContext* context,
        const cpp_llm_engine::cuda::CudaTensor& input,
        cpp_llm_engine::cuda::CudaTensor* output) const;
    cpp_llm_engine::core::Status forward_with_workspace(
        cpp_llm_engine::cuda::CudaExecutionContext* context,
        const cpp_llm_engine::cuda::CudaTensor& input,
        CudaSiglipBlockWorkspace* workspace,
        cpp_llm_engine::cuda::CudaTensor* output) const;

    bool valid() const;
    std::size_t hidden_size() const;
    std::size_t attention_heads() const;

private:
    struct DeviceLinear final {
        cpp_llm_engine::cuda::CudaTensor weight;
        cpp_llm_engine::cuda::CudaQuantizedMatrix quantized_weight;
        cpp_llm_engine::cuda::CudaTensor bias;
        std::size_t input_size = 0;
        std::size_t output_size = 0;
        bool quantized = false;
    };

    static cpp_llm_engine::core::Status upload_linear(
        cpp_llm_engine::cuda::CudaExecutionContext* context,
        const LinearWeights& host,
        DeviceLinear* output);
    static cpp_llm_engine::core::Status upload_gguf_linear(
        cpp_llm_engine::cuda::CudaExecutionContext* context,
        const std::string& path,
        const chapter01::GgufModelIndex& index,
        const std::string& weight_name,
        const std::string& bias_name,
        std::size_t input_size,
        std::size_t output_size,
        DeviceLinear* output);
    static cpp_llm_engine::core::Status run_linear(
        cpp_llm_engine::cuda::CudaExecutionContext* context,
        const cpp_llm_engine::cuda::CudaTensor& input,
        const DeviceLinear& linear,
        cpp_llm_engine::cuda::CudaTensor* output);

    cpp_llm_engine::cuda::CudaTensor norm_1_weight_;
    cpp_llm_engine::cuda::CudaTensor norm_1_bias_;
    DeviceLinear query_;
    DeviceLinear key_;
    DeviceLinear value_;
    DeviceLinear attention_output_;
    cpp_llm_engine::cuda::CudaTensor norm_2_weight_;
    cpp_llm_engine::cuda::CudaTensor norm_2_bias_;
    DeviceLinear feed_forward_input_;
    DeviceLinear feed_forward_output_;
    std::size_t hidden_size_ = 0;
    std::size_t attention_heads_ = 0;
    std::size_t feed_forward_size_ = 0;
    float layer_norm_epsilon_ = 0.0F;
};

}  // namespace multimodal
