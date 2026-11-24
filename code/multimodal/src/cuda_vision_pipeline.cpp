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
    return forward_batch(
        context, {patches}, {positions}, grid_width, grid_height, output);
}

Status CudaSmolvlmVisionPipeline::forward_batch(
    CudaExecutionContext* context,
    const std::vector<PatchTensor>& patch_batch,
    const std::vector<std::vector<std::size_t>>& position_batch,
    std::size_t grid_width,
    std::size_t grid_height,
    CudaTensor* output) {
    if (context == nullptr || output == nullptr || !context->valid() ||
        !valid() || patch_batch.empty() ||
        patch_batch.size() != position_batch.size() ||
        grid_width == 0 || grid_height == 0 ||
        grid_width > std::numeric_limits<std::size_t>::max() / grid_height ||
        patch_batch.size() > std::numeric_limits<std::size_t>::max() /
            (grid_width * grid_height) ||
        patch_batch.size() * grid_width * grid_height >
            static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        return Status::error("CUDA vision pipeline forward input is invalid");
    }
    const std::size_t tokens_per_image = grid_width * grid_height;
    PatchTensor packed;
    packed.patch_count = static_cast<int>(
        patch_batch.size() * tokens_per_image);
    for (std::size_t index = 0; index < patch_batch.size(); ++index) {
        if (!patch_batch[index].valid() ||
            static_cast<std::size_t>(patch_batch[index].patch_count) !=
                tokens_per_image ||
            position_batch[index].size() != tokens_per_image ||
            (index != 0 && patch_batch[index].patch_values !=
                patch_batch.front().patch_values)) {
            return Status::error(
                "CUDA vision pipeline batch shapes are inconsistent");
        }
        packed.values.insert(
            packed.values.end(), patch_batch[index].values.begin(),
            patch_batch[index].values.end());
    }
    packed.patch_values = patch_batch.front().patch_values;
    std::vector<std::size_t> packed_positions;
    packed_positions.reserve(patch_batch.size() * tokens_per_image);
    for (const auto& image_positions : position_batch) {
        packed_positions.insert(
            packed_positions.end(), image_positions.begin(),
            image_positions.end());
    }
    const std::size_t output_tokens = patch_batch.size() *
        (grid_width / merge_size_) * (grid_height / merge_size_);
    if (grid_width % merge_size_ != 0 || grid_height % merge_size_ != 0 ||
        output->contract().shape.dimensions !=
            std::vector<std::size_t>{output_tokens, projection_size_}) {
        return Status::error("CUDA vision pipeline output shape is invalid");
    }

    Status status = prepare_workspace(packed_positions.size());
    if (status.is_ok()) status = embedding_.forward(
        context, packed, packed_positions, &embedded_);
    if (status.is_ok()) status = encoder_.forward_batch(
        context, embedded_, patch_batch.size(), &encoded_);
    if (status.is_ok()) status = projector_.forward_batch(
        context, encoded_, patch_batch.size(), grid_width, grid_height,
        merge_size_, output);
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
