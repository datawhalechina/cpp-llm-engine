// 多模态 CUDA：SigLIP Vision Encoder 设备实现
#include "multimodal/cuda_vision_encoder.h"

#include "cpp_llm_engine/cuda/cuda_execution_context.h"
#include "multimodal/vision_weights.h"

#include <cmath>
#include <utility>

namespace multimodal {
namespace {

using cpp_llm_engine::core::DType;
using cpp_llm_engine::core::Status;
using cpp_llm_engine::cuda::CudaExecutionContext;
using cpp_llm_engine::cuda::CudaTensor;

Status upload_vector(
    CudaExecutionContext* context,
    const std::vector<float>& source,
    CudaTensor* output) {
    Status status = CudaTensor::create_contiguous(
        {source.size()}, DType::Float32, output);
    if (status.is_ok()) {
        status = context->upload(
            output, source.data(), source.size() * sizeof(float));
    }
    return status;
}

}  // namespace

Status CudaSiglipEncoder::upload(
    CudaExecutionContext* context,
    const std::vector<SiglipBlockWeights>& blocks,
    const LayerNormWeights& post_layer_norm,
    std::size_t attention_heads,
    float layer_norm_epsilon,
    CudaSiglipEncoder* output) {
    const std::size_t hidden_size =
        blocks.empty() ? 0 : blocks.front().query.input_size;
    if (context == nullptr || output == nullptr || !context->valid() ||
        blocks.empty() || hidden_size == 0 || attention_heads == 0 ||
        hidden_size % attention_heads != 0 ||
        !post_layer_norm.valid(hidden_size) ||
        !std::isfinite(layer_norm_epsilon) || layer_norm_epsilon <= 0.0F) {
        return Status::error("CUDA SigLIP Encoder upload input is invalid");
    }

    CudaSiglipEncoder replacement;
    replacement.blocks_.reserve(blocks.size());
    for (const SiglipBlockWeights& host_block : blocks) {
        if (!host_block.valid(hidden_size) ||
            host_block.feed_forward_input.output_size !=
                blocks.front().feed_forward_input.output_size) {
            return Status::error("CUDA SigLIP Encoder block shape is inconsistent");
        }
        CudaSiglipBlock device_block;
        Status status = CudaSiglipBlock::upload(
            context, host_block, attention_heads, layer_norm_epsilon,
            &device_block);
        if (!status.is_ok()) return status;
        replacement.blocks_.push_back(std::move(device_block));
    }
    Status status = upload_vector(
        context, post_layer_norm.weight, &replacement.post_norm_weight_);
    if (status.is_ok()) status = upload_vector(
        context, post_layer_norm.bias, &replacement.post_norm_bias_);
    if (!status.is_ok()) return status;
    replacement.hidden_size_ = hidden_size;
    replacement.attention_heads_ = attention_heads;
    replacement.feed_forward_size_ =
        blocks.front().feed_forward_input.output_size;
    replacement.layer_norm_epsilon_ = layer_norm_epsilon;
    *output = std::move(replacement);
    return Status::ok();
}

Status CudaSiglipEncoder::upload_gguf(
    CudaExecutionContext* context,
    const std::string& path,
    const chapter01::GgufModelIndex& index,
    const VisionModelConfig& config,
    const VisionTensorLayout& layout,
    CudaSiglipEncoder* output) {
    if (context == nullptr || output == nullptr || !context->valid() ||
        config.block_count == 0 || layout.blocks.size() != config.block_count) {
        return Status::error("CUDA SigLIP Encoder GGUF input is invalid");
    }
    CudaSiglipEncoder replacement;
    replacement.blocks_.reserve(config.block_count);
    for (std::size_t layer = 0; layer < config.block_count; ++layer) {
        CudaSiglipBlock block;
        Status status = CudaSiglipBlock::upload_gguf(
            context, path, index, config, layout, layer, &block);
        if (!status.is_ok()) return status;
        replacement.blocks_.push_back(std::move(block));
    }
    LayerNormWeights post_norm;
    std::string error;
    if (!load_post_layer_norm_weights(
            path, index, config, layout, &post_norm, &error)) {
        return Status::error(error);
    }
    Status status = upload_vector(
        context, post_norm.weight, &replacement.post_norm_weight_);
    if (status.is_ok()) status = upload_vector(
        context, post_norm.bias, &replacement.post_norm_bias_);
    if (!status.is_ok()) return status;
    replacement.hidden_size_ = config.hidden_size;
    replacement.attention_heads_ = config.attention_head_count;
    replacement.feed_forward_size_ = config.intermediate_size;
    replacement.layer_norm_epsilon_ = config.layer_norm_epsilon;
    *output = std::move(replacement);
    return Status::ok();
}

Status CudaSiglipEncoder::prepare_workspace(
    CudaExecutionContext* context,
    std::size_t token_count) {
    if (prepared_tokens_ == token_count) return Status::ok();
    if (prepared_tokens_ != 0) {
        Status status = context->synchronize();
        if (!status.is_ok()) return status;
    }
    CudaSiglipBlockWorkspace workspace;
    CudaTensor hidden_a;
    CudaTensor hidden_b;
    Status status = CudaSiglipBlockWorkspace::create(
        token_count, hidden_size_, attention_heads_, feed_forward_size_,
        &workspace);
    if (status.is_ok()) status = CudaTensor::create_contiguous(
        {token_count, hidden_size_}, DType::Float32, &hidden_a);
    if (status.is_ok()) status = CudaTensor::create_contiguous(
        {token_count, hidden_size_}, DType::Float32, &hidden_b);
    if (!status.is_ok()) return status;
    workspace_ = std::move(workspace);
    hidden_a_ = std::move(hidden_a);
    hidden_b_ = std::move(hidden_b);
    prepared_tokens_ = token_count;
    ++workspace_rebuild_count_;
    return Status::ok();
}

Status CudaSiglipEncoder::forward(
    CudaExecutionContext* context,
    const CudaTensor& input,
    CudaTensor* output) {
    return forward_batch(context, input, 1, output);
}

Status CudaSiglipEncoder::forward_batch(
    CudaExecutionContext* context,
    const CudaTensor& input,
    std::size_t batch_size,
    CudaTensor* output) {
    if (context == nullptr || output == nullptr || !context->valid() ||
        !valid() || batch_size == 0) {
        return Status::error("CUDA SigLIP Encoder forward input is invalid");
    }
    const auto& shape = input.contract().shape.dimensions;
    if (shape.size() != 2 || shape[0] == 0 ||
        shape[0] % batch_size != 0 || shape[1] != hidden_size_ ||
        output->contract().shape.dimensions != shape) {
        return Status::error("CUDA SigLIP Encoder input/output shape is invalid");
    }
    Status status = prepare_workspace(context, shape[0]);
    if (!status.is_ok()) return status;

    const CudaTensor* current = &input;
    for (std::size_t layer = 0; layer < blocks_.size(); ++layer) {
        CudaTensor* destination = layer % 2 == 0 ? &hidden_a_ : &hidden_b_;
        status = blocks_[layer].forward_batch_with_workspace(
            context, *current, batch_size, &workspace_, destination);
        if (!status.is_ok()) return status;
        current = destination;
    }
    return context->layer_norm_batch(
        *current, post_norm_weight_, post_norm_bias_, output,
        layer_norm_epsilon_);
}

bool CudaSiglipEncoder::valid() const {
    return !blocks_.empty() && hidden_size_ > 0 && attention_heads_ > 0 &&
        hidden_size_ % attention_heads_ == 0 && feed_forward_size_ > 0 &&
        !post_norm_weight_.empty() && !post_norm_bias_.empty() &&
        std::isfinite(layer_norm_epsilon_) && layer_norm_epsilon_ > 0.0F;
}

std::size_t CudaSiglipEncoder::block_count() const { return blocks_.size(); }
std::size_t CudaSiglipEncoder::workspace_rebuild_count() const {
    return workspace_rebuild_count_;
}

}  // namespace multimodal
