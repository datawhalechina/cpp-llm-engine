// 多模态 CUDA：SmolVLM 视觉设备流水线实现
#include "multimodal/cuda_vision_pipeline.h"

#include "cpp_llm_engine/cuda/cuda_execution_context.h"
#include "multimodal/vision_weights.h"

#include <limits>
#include <utility>

namespace multimodal {
namespace {

using cpp_llm_engine::core::DType;
using cpp_llm_engine::core::Status;
using cpp_llm_engine::cuda::CudaExecutionContext;
using cpp_llm_engine::cuda::CudaTensor;

}  // namespace

Status CudaSmolvlmVisionPipeline::upload_gguf(
    CudaExecutionContext* context,
    const std::string& path,
    const chapter01::GgufModelIndex& index,
    const VisionModelConfig& config,
    const VisionTensorLayout& layout,
    CudaSmolvlmVisionPipeline* output) {
    if (context == nullptr || output == nullptr || !context->valid() ||
        path.empty() || config.merge_size == 0 || config.projection_dim == 0) {
        return Status::error("CUDA vision pipeline upload input is invalid");
    }

    PatchEmbeddingWeights patch_weights;
    LinearWeights projector_weights;
    std::string error;
    if (!load_patch_embedding_weights(
            path, index, config, layout, &patch_weights, &error) ||
        !load_idefics3_projector_weights(
            path, index, config, layout, &projector_weights, &error)) {
        return Status::error("CUDA vision pipeline weight load failed: " + error);
    }

    CudaSmolvlmVisionPipeline replacement;
    Status status = CudaPatchEmbedding::upload(
        context, patch_weights, &replacement.embedding_);
    if (status.is_ok()) status = CudaSiglipEncoder::upload_gguf(
        context, path, index, config, layout, &replacement.encoder_);
    if (status.is_ok()) status = CudaIdefics3Projector::upload(
        context, projector_weights, &replacement.projector_);
    if (!status.is_ok()) return status;

    replacement.merge_size_ = config.merge_size;
    replacement.projection_size_ = config.projection_dim;
    *output = std::move(replacement);
    return Status::ok();
}

Status CudaSmolvlmVisionPipeline::prepare_workspace(
    std::size_t token_count) {
    if (token_count == prepared_tokens_) return Status::ok();
    CudaTensor embedded;
    CudaTensor encoded;
    Status status = CudaTensor::create_contiguous(
        {token_count, embedding_.hidden_size()}, DType::Float32, &embedded);
    if (status.is_ok()) status = CudaTensor::create_contiguous(
        {token_count, embedding_.hidden_size()}, DType::Float32, &encoded);
    if (!status.is_ok()) return status;
    embedded_ = std::move(embedded);
    encoded_ = std::move(encoded);
    prepared_tokens_ = token_count;
    ++workspace_rebuild_count_;
    return Status::ok();
}

Status CudaSmolvlmVisionPipeline::forward(
    CudaExecutionContext* context,
    const PatchTensor& patches,
    const std::vector<std::size_t>& positions,
    std::size_t grid_width,
    std::size_t grid_height,
    CudaTensor* output) {
    if (context == nullptr || output == nullptr || !context->valid() ||
        !valid() || !patches.valid() || grid_width == 0 || grid_height == 0 ||
        grid_width > std::numeric_limits<std::size_t>::max() / grid_height ||
        grid_width * grid_height != positions.size() ||
        positions.size() != static_cast<std::size_t>(patches.patch_count)) {
        return Status::error("CUDA vision pipeline forward input is invalid");
    }
    const std::size_t output_tokens =
        (grid_width / merge_size_) * (grid_height / merge_size_);
    if (grid_width % merge_size_ != 0 || grid_height % merge_size_ != 0 ||
        output->contract().shape.dimensions !=
            std::vector<std::size_t>{output_tokens, projection_size_}) {
        return Status::error("CUDA vision pipeline output shape is invalid");
    }

    Status status = prepare_workspace(positions.size());
    if (status.is_ok()) status = embedding_.forward(
        context, patches, positions, &embedded_);
    if (status.is_ok()) status = encoder_.forward(
        context, embedded_, &encoded_);
    if (status.is_ok()) status = projector_.forward(
        context, encoded_, grid_width, grid_height, merge_size_, output);
    return status;
}

bool CudaSmolvlmVisionPipeline::valid() const {
    return embedding_.valid() && encoder_.valid() && projector_.valid() &&
        merge_size_ > 0 && projection_size_ == projector_.output_size();
}

std::size_t CudaSmolvlmVisionPipeline::workspace_rebuild_count() const {
    return workspace_rebuild_count_;
}

}  // namespace multimodal
