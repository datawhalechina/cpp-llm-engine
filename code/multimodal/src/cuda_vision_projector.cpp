// 多模态 CUDA：IDEFICS3 空间合并与 Projector 设备实现
#include "multimodal/cuda_vision_projector.h"

#include "cpp_llm_engine/cuda/cuda_execution_context.h"
#include "cpp_llm_engine/cuda/cuda_linear_weight.h"

#include <limits>
#include <utility>
#include <vector>

namespace multimodal {
namespace {

using cpp_llm_engine::core::DType;
using cpp_llm_engine::core::Status;
using cpp_llm_engine::cuda::CudaExecutionContext;
using cpp_llm_engine::cuda::CudaLinearWeight;
using cpp_llm_engine::cuda::CudaTensor;

}  // namespace

Status CudaIdefics3Projector::upload(
    CudaExecutionContext* context,
    const LinearWeights& host,
    CudaIdefics3Projector* output) {
    if (context == nullptr || output == nullptr || !context->valid() ||
        !host.valid() || !host.bias.empty()) {
        return Status::error("CUDA IDEFICS3 projector upload input is invalid");
    }

    std::vector<float> transposed(host.input_size * host.output_size);
    for (std::size_t destination = 0; destination < host.output_size;
         ++destination) {
        for (std::size_t source = 0; source < host.input_size; ++source) {
            transposed[source * host.output_size + destination] =
                host.weight[destination * host.input_size + source];
        }
    }

    CudaIdefics3Projector replacement;
    Status status = CudaTensor::create_contiguous(
        {host.input_size, host.output_size}, DType::Float32,
        &replacement.weight_);
    if (status.is_ok()) {
        status = context->upload(
            &replacement.weight_, transposed.data(),
            transposed.size() * sizeof(float));
    }
    if (!status.is_ok()) return status;
    replacement.input_size_ = host.input_size;
    replacement.output_size_ = host.output_size;
    *output = std::move(replacement);
    return Status::ok();
}

Status CudaIdefics3Projector::forward(
    CudaExecutionContext* context,
    const CudaTensor& input,
    std::size_t grid_width,
    std::size_t grid_height,
    std::size_t merge_size,
    CudaTensor* output) const {
    return forward_batch(
        context, input, 1, grid_width, grid_height, merge_size, output);
}

Status CudaIdefics3Projector::forward_batch(
    CudaExecutionContext* context,
    const CudaTensor& input,
    std::size_t batch_size,
    std::size_t grid_width,
    std::size_t grid_height,
    std::size_t merge_size,
    CudaTensor* output) const {
    if (context == nullptr || output == nullptr || !context->valid() ||
        !valid() || batch_size == 0 || grid_width == 0 || grid_height == 0 ||
        merge_size == 0 ||
        grid_width % merge_size != 0 || grid_height % merge_size != 0) {
        return Status::error("CUDA IDEFICS3 projector forward input is invalid");
    }
    const auto& input_shape = input.contract().shape.dimensions;
    if (grid_width > std::numeric_limits<std::size_t>::max() / grid_height ||
        batch_size > std::numeric_limits<std::size_t>::max() /
            (grid_width * grid_height) || input_shape.size() != 2 ||
        input_shape[0] != batch_size * grid_width * grid_height ||
        merge_size > std::numeric_limits<std::size_t>::max() / merge_size ||
        merge_size * merge_size >
            std::numeric_limits<std::size_t>::max() / input_shape[1] ||
        merge_size * merge_size * input_shape[1] != input_size_) {
        return Status::error("CUDA IDEFICS3 projector shape is invalid");
    }
    const std::size_t output_tokens =
        batch_size * (grid_width / merge_size) *
        (grid_height / merge_size);
    if (output->empty() ||
        output->contract().dtype != DType::Float32 ||
        output->contract().device != cpp_llm_engine::core::DeviceKind::CUDA ||
        output->contract().shape.dimensions !=
        std::vector<std::size_t>{output_tokens, output_size_}) {
        return Status::error("CUDA IDEFICS3 projector output shape is invalid");
    }

    CudaTensor merged;
    Status status = CudaTensor::create_contiguous(
        {output_tokens, input_size_}, DType::Float32, &merged);
    if (status.is_ok()) {
        status = context->spatial_merge_batch(
            input, batch_size, grid_width, grid_height, merge_size, &merged);
    }
    if (status.is_ok()) {
        status = context->linear_batch(
            merged, CudaLinearWeight::float32(&weight_), output);
    }
    // The merged scratch tensor must outlive all work queued on the CUDA stream.
    if (status.is_ok()) status = context->synchronize();
    return status;
}

bool CudaIdefics3Projector::valid() const {
    return input_size_ > 0 && output_size_ > 0 &&
        weight_.contract().shape.dimensions ==
            std::vector<std::size_t>{input_size_, output_size_};
}

std::size_t CudaIdefics3Projector::input_size() const { return input_size_; }
std::size_t CudaIdefics3Projector::output_size() const { return output_size_; }

}  // namespace multimodal
