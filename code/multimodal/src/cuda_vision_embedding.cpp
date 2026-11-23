// 多模态 CUDA：Patch/Position Embedding 设备适配实现
#include "multimodal/cuda_vision_embedding.h"

#include "cpp_llm_engine/cuda/cuda_execution_context.h"
#include "cpp_llm_engine/cuda/cuda_linear_weight.h"

#include <utility>

namespace multimodal {
namespace {

using cpp_llm_engine::core::DType;
using cpp_llm_engine::core::Status;
using cpp_llm_engine::cuda::CudaExecutionContext;
using cpp_llm_engine::cuda::CudaLinearWeight;
using cpp_llm_engine::cuda::CudaTensor;

}  // namespace

Status CudaPatchEmbedding::upload(
    CudaExecutionContext* context,
    const PatchEmbeddingWeights& host,
    CudaPatchEmbedding* output) {
    if (context == nullptr || output == nullptr || !context->valid() ||
        !host.valid()) {
        return Status::error("CUDA patch embedding upload input is invalid");
    }

    CudaPatchEmbedding replacement;
    std::vector<float> transposed(
        host.patch_values * host.hidden_size);
    for (std::size_t hidden = 0; hidden < host.hidden_size; ++hidden) {
        for (std::size_t input = 0; input < host.patch_values; ++input) {
            transposed[input * host.hidden_size + hidden] =
                host.patch_weight[hidden * host.patch_values + input];
        }
    }
    std::vector<float> position_and_bias = host.position_weight;
    for (std::size_t position = 0; position < host.position_count; ++position) {
        for (std::size_t hidden = 0; hidden < host.hidden_size; ++hidden) {
            position_and_bias[position * host.hidden_size + hidden] +=
                host.patch_bias[hidden];
        }
    }

    Status status = CudaTensor::create_contiguous(
        {host.patch_values, host.hidden_size}, DType::Float32,
        &replacement.patch_weight_);
    if (status.is_ok()) status = CudaTensor::create_contiguous(
        {host.position_count, host.hidden_size}, DType::Float32,
        &replacement.position_and_bias_);
    if (status.is_ok()) status = context->upload(
        &replacement.patch_weight_, transposed.data(),
        transposed.size() * sizeof(float));
    if (status.is_ok()) status = context->upload(
        &replacement.position_and_bias_, position_and_bias.data(),
        position_and_bias.size() * sizeof(float));
    if (!status.is_ok()) return status;

    replacement.patch_values_ = host.patch_values;
    replacement.hidden_size_ = host.hidden_size;
    replacement.position_count_ = host.position_count;
    *output = std::move(replacement);
    return Status::ok();
}

Status CudaPatchEmbedding::forward(
    CudaExecutionContext* context,
    const PatchTensor& patches,
    const std::vector<std::size_t>& positions,
    CudaTensor* output) const {
    if (context == nullptr || output == nullptr || !context->valid() ||
        !valid() || !patches.valid() ||
        static_cast<std::size_t>(patches.patch_values) != patch_values_ ||
        positions.size() != static_cast<std::size_t>(patches.patch_count) ||
        output->contract().shape.dimensions !=
            std::vector<std::size_t>{positions.size(), hidden_size_}) {
        return Status::error("CUDA patch embedding forward input is invalid");
    }
    for (const std::size_t position : positions) {
        if (position >= position_count_) {
            return Status::error("CUDA patch position is out of range");
        }
    }

    CudaTensor patch_input;
    CudaTensor selected_position;
    Status status = CudaTensor::create_contiguous(
        {positions.size(), patch_values_}, DType::Float32, &patch_input);
    if (status.is_ok()) status = CudaTensor::create_contiguous(
        {positions.size(), hidden_size_}, DType::Float32,
        &selected_position);
    if (status.is_ok()) status = context->upload(
        &patch_input, patches.values.data(),
        patches.values.size() * sizeof(float));
    if (status.is_ok()) status = context->gather_rows(
        position_and_bias_, positions, &selected_position);
    if (status.is_ok()) status = context->linear_batch(
        patch_input, CudaLinearWeight::float32(&patch_weight_), output);
    if (status.is_ok()) status = context->vector_add(
        *output, selected_position, output);
    // Local scratch tensors must outlive all work queued on the CUDA stream.
    if (status.is_ok()) status = context->synchronize();
    return status;
}

bool CudaPatchEmbedding::valid() const {
    return patch_values_ > 0 && hidden_size_ > 0 && position_count_ > 0 &&
        patch_weight_.contract().shape.dimensions ==
            std::vector<std::size_t>{patch_values_, hidden_size_} &&
        position_and_bias_.contract().shape.dimensions ==
            std::vector<std::size_t>{position_count_, hidden_size_};
}

std::size_t CudaPatchEmbedding::patch_values() const { return patch_values_; }
std::size_t CudaPatchEmbedding::hidden_size() const { return hidden_size_; }
std::size_t CudaPatchEmbedding::position_count() const {
    return position_count_;
}

}  // namespace multimodal
