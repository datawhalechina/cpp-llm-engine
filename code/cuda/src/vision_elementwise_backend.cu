// CUDA Backend：实现标准 LayerNorm、最后一维 bias 广播与 tanh GELU。
#include "cpp_llm_engine/cuda/vision_elementwise_backend.h"

#include "cpp_llm_engine/cuda/cuda_stream.h"

#include <cuda_runtime.h>

#include <cmath>
#include <limits>

namespace cpp_llm_engine::cuda {
namespace {

constexpr unsigned int kBlockSize = 256;

__global__ void layer_norm_batch_kernel(CudaLayerNormBatchRequest request) {
    const std::size_t row = blockIdx.x;
    if (row >= request.rows) return;
    __shared__ float sums[kBlockSize];
    __shared__ float squares[kBlockSize];
    const std::size_t offset = row * request.width;
    float local_sum = 0.0F;
    float local_square = 0.0F;
    for (std::size_t col = threadIdx.x; col < request.width;
         col += blockDim.x) {
        const float value = request.input[offset + col];
        local_sum += value;
        local_square += value * value;
    }
    sums[threadIdx.x] = local_sum;
    squares[threadIdx.x] = local_square;
    __syncthreads();
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            sums[threadIdx.x] += sums[threadIdx.x + stride];
            squares[threadIdx.x] += squares[threadIdx.x + stride];
        }
        __syncthreads();
    }
    const float mean = sums[0] / static_cast<float>(request.width);
    const float variance =
        squares[0] / static_cast<float>(request.width) - mean * mean;
    const float inverse = rsqrtf(fmaxf(variance, 0.0F) + request.epsilon);
    for (std::size_t col = threadIdx.x; col < request.width;
         col += blockDim.x) {
        request.output[offset + col] =
            (request.input[offset + col] - mean) * inverse *
                request.weight[col] + request.bias[col];
    }
}

__global__ void bias_batch_kernel(CudaBiasBatchRequest request) {
    const std::size_t index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= request.elements) return;
    request.output[index] = request.input[index] + request.bias[index % request.width];
}

__global__ void gelu_kernel(CudaGeluRequest request) {
    const std::size_t index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= request.elements) return;
    constexpr float kSqrtTwoOverPi = 0.7978845608028654F;
    const float value = request.input[index];
    request.output[index] = 0.5F * value *
        (1.0F + tanhf(kSqrtTwoOverPi *
            (value + 0.044715F * value * value * value)));
}

bool valid_grid(std::size_t elements) {
    const std::size_t blocks = elements / kBlockSize +
        (elements % kBlockSize != 0 ? 1 : 0);
    return blocks <= std::numeric_limits<unsigned int>::max();
}

}  // namespace

CudaStatus launch_layer_norm_batch(
    const CudaLayerNormBatchRequest& request, const CudaStream& stream) {
    if (request.input == nullptr || request.weight == nullptr ||
        request.bias == nullptr || request.output == nullptr ||
        request.rows == 0 || request.width == 0 ||
        !std::isfinite(request.epsilon) || request.epsilon <= 0.0F ||
        request.rows > std::numeric_limits<unsigned int>::max() ||
        !stream.valid()) {
        return CudaStatus::from(cudaErrorInvalidValue, "vision LayerNorm request");
    }
    layer_norm_batch_kernel<<<static_cast<unsigned int>(request.rows),
        kBlockSize, 0, stream.native_handle()>>>(request);
    return check_last_cuda_error("vision_layer_norm_batch_kernel launch");
}

CudaStatus launch_bias_batch(
    const CudaBiasBatchRequest& request, const CudaStream& stream) {
    if (request.input == nullptr || request.bias == nullptr ||
        request.output == nullptr || request.elements == 0 ||
        request.width == 0 || request.elements % request.width != 0 ||
        !valid_grid(request.elements) || !stream.valid()) {
        return CudaStatus::from(cudaErrorInvalidValue, "vision bias request");
    }
    const std::size_t blocks = request.elements / kBlockSize +
        (request.elements % kBlockSize != 0 ? 1 : 0);
    bias_batch_kernel<<<static_cast<unsigned int>(blocks), kBlockSize, 0,
        stream.native_handle()>>>(request);
    return check_last_cuda_error("vision_bias_batch_kernel launch");
}

CudaStatus launch_gelu(
    const CudaGeluRequest& request, const CudaStream& stream) {
    if (request.input == nullptr || request.output == nullptr ||
        request.elements == 0 || !valid_grid(request.elements) ||
        !stream.valid()) {
        return CudaStatus::from(cudaErrorInvalidValue, "vision GELU request");
    }
    const std::size_t blocks = request.elements / kBlockSize +
        (request.elements % kBlockSize != 0 ? 1 : 0);
    gelu_kernel<<<static_cast<unsigned int>(blocks), kBlockSize, 0,
        stream.native_handle()>>>(request);
    return check_last_cuda_error("vision_gelu_kernel launch");
}

}  // namespace cpp_llm_engine::cuda
