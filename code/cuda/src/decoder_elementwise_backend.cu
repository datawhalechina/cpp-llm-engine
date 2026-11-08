// CUDA Backend：实现单向量 RMSNorm 与 SwiGLU。
// 契约：RMSNorm 用单 block 归约平方和；SwiGLU 每线程处理一个元素。
// 验证：cpp_llm_engine_cuda_decoder_elementwise_test 与 CPU 公式对照。
#include "cpp_llm_engine/cuda/decoder_elementwise_backend.h"

#include "cpp_llm_engine/cuda/cuda_stream.h"

#include <cuda_runtime.h>

namespace cpp_llm_engine::cuda {
namespace {

constexpr unsigned int kBlockSize = 256;

__global__ void rms_norm_kernel(
    const float* input, const float* weight, float* output,
    std::size_t elements, float epsilon) {
    __shared__ float squares[kBlockSize];
    float local = 0.0F;
    for (std::size_t index = threadIdx.x; index < elements; index += blockDim.x) {
        local += input[index] * input[index];
    }
    squares[threadIdx.x] = local;
    __syncthreads();
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) squares[threadIdx.x] += squares[threadIdx.x + stride];
        __syncthreads();
    }
    const float inverse = rsqrtf(squares[0] / static_cast<float>(elements) + epsilon);
    for (std::size_t index = threadIdx.x; index < elements; index += blockDim.x) {
        output[index] = input[index] * inverse * weight[index];
    }
}

__global__ void rms_norm_batch_kernel(
    const float* input, const float* weight, float* output,
    std::size_t rows, std::size_t width, float epsilon) {
    const std::size_t row = blockIdx.x;
    if (row >= rows) return;
    __shared__ float squares[kBlockSize];
    const std::size_t offset = row * width;
    float local = 0.0F;
    for (std::size_t col = threadIdx.x; col < width; col += blockDim.x) {
        const float value = input[offset + col];
        local += value * value;
    }
    squares[threadIdx.x] = local;
    __syncthreads();
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) squares[threadIdx.x] += squares[threadIdx.x + stride];
        __syncthreads();
    }
    const float inverse = rsqrtf(squares[0] / static_cast<float>(width) + epsilon);
    for (std::size_t col = threadIdx.x; col < width; col += blockDim.x) {
        output[offset + col] = input[offset + col] * inverse * weight[col];
    }
}

__global__ void swiglu_kernel(
    const float* gate, const float* up, float* output, std::size_t elements) {
    const std::size_t index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= elements) return;
    const float sigmoid = 1.0F / (1.0F + expf(-gate[index]));
    output[index] = gate[index] * sigmoid * up[index];
}

}  // namespace

core::BackendKind CudaDecoderElementwiseBackend::kind() const {
    return core::BackendKind::CUDA;
}

std::string_view CudaDecoderElementwiseBackend::name() const {
    return "cuda-decoder-elementwise";
}

CudaStatus CudaDecoderElementwiseBackend::rms_norm(
    const core::RmsNormRequest& request, const CudaStream& stream) const {
    if (!request.validate().is_ok()) {
        return CudaStatus::from(cudaErrorInvalidValue, "CUDA RMSNorm request");
    }
    if (!stream.valid()) {
        return CudaStatus::from(cudaErrorInvalidResourceHandle, "CUDA RMSNorm stream");
    }
    rms_norm_kernel<<<1, kBlockSize, 0, stream.native_handle()>>>(
        request.input, request.weight, request.output, request.elements,
        request.epsilon);
    return check_last_cuda_error("rms_norm_kernel launch");
}

CudaStatus CudaDecoderElementwiseBackend::rms_norm_batch(
    const core::RmsNormBatchRequest& request,
    const CudaStream& stream) const {
    if (!request.validate().is_ok()) {
        return CudaStatus::from(cudaErrorInvalidValue, "CUDA batch RMSNorm request");
    }
    if (!stream.valid()) {
        return CudaStatus::from(
            cudaErrorInvalidResourceHandle, "CUDA batch RMSNorm stream");
    }
    rms_norm_batch_kernel<<<static_cast<unsigned int>(request.rows), kBlockSize, 0,
                            stream.native_handle()>>>(
        request.input, request.weight, request.output,
        request.rows, request.width, request.epsilon);
    return check_last_cuda_error("rms_norm_batch_kernel launch");
}

CudaStatus CudaDecoderElementwiseBackend::swiglu(
    const core::SwiGluRequest& request, const CudaStream& stream) const {
    if (!request.validate().is_ok()) {
        return CudaStatus::from(cudaErrorInvalidValue, "CUDA SwiGLU request");
    }
    if (!stream.valid()) {
        return CudaStatus::from(cudaErrorInvalidResourceHandle, "CUDA SwiGLU stream");
    }
    const std::size_t blocks = request.elements / kBlockSize +
        (request.elements % kBlockSize != 0 ? 1 : 0);
    swiglu_kernel<<<static_cast<unsigned int>(blocks), kBlockSize, 0,
                    stream.native_handle()>>>(
        request.gate, request.up, request.output, request.elements);
    return check_last_cuda_error("swiglu_kernel launch");
}

const CudaDecoderElementwiseBackend& cuda_decoder_elementwise_backend() {
    static const CudaDecoderElementwiseBackend backend;
    return backend;
}

}  // namespace cpp_llm_engine::cuda
