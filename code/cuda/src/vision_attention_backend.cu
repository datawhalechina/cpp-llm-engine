// CUDA Backend：实现按图片边界隔离的三阶段视觉注意力。
#include "cpp_llm_engine/cuda/vision_attention_backend.h"

#include "cpp_llm_engine/cuda/cuda_stream.h"

#include <cuda_runtime.h>

#include <cmath>
#include <limits>

namespace cpp_llm_engine::cuda {
namespace {

constexpr std::size_t kBlockSize = 256;

__global__ void vision_attention_score_kernel(
    CudaVisionAttentionRequest request, float* scores) {
    const std::size_t index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t score_count = request.batch_size *
        request.tokens_per_image * request.head_count *
        request.tokens_per_image;
    if (index >= score_count) return;
    const std::size_t key_token = index % request.tokens_per_image;
    const std::size_t row = index / request.tokens_per_image;
    const std::size_t head = row % request.head_count;
    const std::size_t packed_query = row / request.head_count;
    const std::size_t image = packed_query / request.tokens_per_image;
    const std::size_t query_token = packed_query % request.tokens_per_image;
    const std::size_t hidden = request.head_count * request.head_dimension;
    const std::size_t query_offset =
        (image * request.tokens_per_image + query_token) * hidden +
        head * request.head_dimension;
    const std::size_t key_offset =
        (image * request.tokens_per_image + key_token) * hidden +
        head * request.head_dimension;
    const float scale = request.scale == 0.0F
        ? rsqrtf(static_cast<float>(request.head_dimension))
        : request.scale;
    float score = 0.0F;
    for (std::size_t dimension = 0; dimension < request.head_dimension;
         ++dimension) {
        score += request.query[query_offset + dimension] *
            request.key[key_offset + dimension];
    }
    scores[index] = score * scale;
}

__global__ void vision_attention_softmax_kernel(
    float* scores, std::size_t rows, std::size_t width) {
    const std::size_t row =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    float maximum = -INFINITY;
    for (std::size_t column = 0; column < width; ++column) {
        maximum = fmaxf(maximum, scores[row * width + column]);
    }
    float sum = 0.0F;
    for (std::size_t column = 0; column < width; ++column) {
        const float value = expf(scores[row * width + column] - maximum);
        scores[row * width + column] = value;
        sum += value;
    }
    for (std::size_t column = 0; column < width; ++column) {
        scores[row * width + column] /= sum;
    }
}

__global__ void vision_attention_value_kernel(
    CudaVisionAttentionRequest request, const float* scores) {
    const std::size_t index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t output_elements = request.batch_size *
        request.tokens_per_image * request.head_count * request.head_dimension;
    if (index >= output_elements) return;
    const std::size_t dimension = index % request.head_dimension;
    const std::size_t head_row = index / request.head_dimension;
    const std::size_t head = head_row % request.head_count;
    const std::size_t packed_query = head_row / request.head_count;
    const std::size_t image = packed_query / request.tokens_per_image;
    const std::size_t hidden = request.head_count * request.head_dimension;
    float result = 0.0F;
    for (std::size_t key_token = 0; key_token < request.tokens_per_image;
         ++key_token) {
        const std::size_t value_offset =
            (image * request.tokens_per_image + key_token) * hidden +
            head * request.head_dimension + dimension;
        result += scores[head_row * request.tokens_per_image + key_token] *
            request.value[value_offset];
    }
    request.output[index] = result;
}

bool valid_grid(std::size_t elements) {
    const std::size_t blocks = elements / kBlockSize +
        (elements % kBlockSize != 0 ? 1 : 0);
    return blocks <= std::numeric_limits<unsigned int>::max();
}

}  // namespace

std::size_t CudaVisionAttentionRequest::score_count() const {
    return batch_size * tokens_per_image * head_count * tokens_per_image;
}

CudaStatus launch_vision_attention(
    const CudaVisionAttentionRequest& request,
    float* score_workspace,
    std::size_t workspace_elements,
    const CudaStream& stream) {
    const std::size_t rows =
        request.batch_size * request.tokens_per_image * request.head_count;
    const std::size_t output_elements = rows * request.head_dimension;
    if (request.query == nullptr || request.key == nullptr ||
        request.value == nullptr || request.output == nullptr ||
        score_workspace == nullptr || request.batch_size == 0 ||
        request.tokens_per_image == 0 || request.head_count == 0 ||
        request.head_dimension == 0 ||
        workspace_elements < request.score_count() ||
        !valid_grid(request.score_count()) || !valid_grid(rows) ||
        !valid_grid(output_elements) || !stream.valid()) {
        return CudaStatus::from(
            cudaErrorInvalidValue, "vision attention request");
    }
    const auto blocks_for = [](std::size_t elements) {
        return static_cast<unsigned int>(elements / kBlockSize +
            (elements % kBlockSize != 0 ? 1 : 0));
    };
    vision_attention_score_kernel<<<blocks_for(request.score_count()),
        kBlockSize, 0, stream.native_handle()>>>(request, score_workspace);
    CudaStatus status = check_last_cuda_error(
        "vision_attention_score_kernel launch");
    if (!status.is_ok()) return status;
    vision_attention_softmax_kernel<<<blocks_for(rows), kBlockSize, 0,
        stream.native_handle()>>>(
            score_workspace, rows, request.tokens_per_image);
    status = check_last_cuda_error("vision_attention_softmax_kernel launch");
    if (!status.is_ok()) return status;
    vision_attention_value_kernel<<<blocks_for(output_elements), kBlockSize, 0,
        stream.native_handle()>>>(request, score_workspace);
    return check_last_cuda_error("vision_attention_value_kernel launch");
}

}  // namespace cpp_llm_engine::cuda
