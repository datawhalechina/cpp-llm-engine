// CUDA Backend：实现原地 RoPE pair kernel 与 launch 参数检查。
// 契约：每个线程旋转一个分量对，支持相邻维和 NeoX 半维配对。
// 验证：cpp_llm_engine_cuda_rope_test 与两种 CPU 公式逐元素对照。
#include "cpp_llm_engine/cuda/rope_backend.h"

#include "cpp_llm_engine/cuda/cuda_stream.h"

#include <cuda_runtime.h>

#include <limits>

namespace cpp_llm_engine::cuda {
namespace {

constexpr std::size_t kBlockSize = 256;

__global__ void rope_kernel(
    float* values,
    std::size_t pair_count,
    std::size_t head_count,
    std::size_t head_dimension,
    std::size_t rotary_dimension,
    std::size_t position_offset,
    float frequency_base,
    core::RotaryPairing pairing) {
    const std::size_t index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= pair_count) return;

    const std::size_t pairs_per_head = rotary_dimension / 2;
    const std::size_t pair = index % pairs_per_head;
    const std::size_t token_head = index / pairs_per_head;
    const std::size_t head = token_head % head_count;
    const std::size_t token = token_head / head_count;
    const std::size_t vector_base =
        (token * head_count + head) * head_dimension;
    const std::size_t first = pairing == core::RotaryPairing::SplitHalf
        ? pair : pair * 2;
    const std::size_t second = pairing == core::RotaryPairing::SplitHalf
        ? pair + pairs_per_head : pair * 2 + 1;
    const float exponent =
        -2.0F * static_cast<float>(pair) / static_cast<float>(rotary_dimension);
    const float angle = static_cast<float>(position_offset + token) *
                        powf(frequency_base, exponent);
    float sine = 0.0F;
    float cosine = 0.0F;
    sincosf(angle, &sine, &cosine);
    const float x = values[vector_base + first];
    const float y = values[vector_base + second];
    values[vector_base + first] = x * cosine - y * sine;
    values[vector_base + second] = x * sine + y * cosine;
}

__global__ void rope_batch_kernel(
    float* values,
    const std::size_t* positions,
    std::size_t pair_count,
    std::size_t head_count,
    std::size_t head_dimension,
    std::size_t rotary_dimension,
    float frequency_base,
    core::RotaryPairing pairing) {
    const std::size_t index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= pair_count) return;

    const std::size_t pairs_per_head = rotary_dimension / 2;
    const std::size_t pair = index % pairs_per_head;
    const std::size_t row_head = index / pairs_per_head;
    const std::size_t head = row_head % head_count;
    const std::size_t row = row_head / head_count;
    const std::size_t vector_base =
        (row * head_count + head) * head_dimension;
    const std::size_t first = pairing == core::RotaryPairing::SplitHalf
        ? pair : pair * 2;
    const std::size_t second = pairing == core::RotaryPairing::SplitHalf
        ? pair + pairs_per_head : pair * 2 + 1;
    const float exponent =
        -2.0F * static_cast<float>(pair) / static_cast<float>(rotary_dimension);
    const float angle = static_cast<float>(positions[row]) *
                        powf(frequency_base, exponent);
    float sine = 0.0F;
    float cosine = 0.0F;
    sincosf(angle, &sine, &cosine);
    const float x = values[vector_base + first];
    const float y = values[vector_base + second];
    values[vector_base + first] = x * cosine - y * sine;
    values[vector_base + second] = x * sine + y * cosine;
}

}  // namespace

core::BackendKind CudaRopeBackend::kind() const {
    return core::BackendKind::CUDA;
}

std::string_view CudaRopeBackend::name() const {
    return "cuda-rope";
}

CudaStatus CudaRopeBackend::apply(
    const core::RotaryEmbeddingRequest& request,
    const CudaStream& stream) const {
    if (request.token_count == 0) return CudaStatus::success();
    if (!request.validate().is_ok()) {
        return CudaStatus::from(cudaErrorInvalidValue, "CudaRopeBackend::apply request");
    }
    if (!stream.valid()) {
        return CudaStatus::from(
            cudaErrorInvalidResourceHandle, "CudaRopeBackend::apply stream");
    }
    const std::size_t pairs_per_head = request.rotary_dimension / 2;
    if (request.token_count > std::numeric_limits<std::size_t>::max() /
                                  request.head_count ||
        request.token_count * request.head_count >
            std::numeric_limits<std::size_t>::max() / pairs_per_head) {
        return CudaStatus::from(
            cudaErrorInvalidConfiguration, "CudaRopeBackend::apply pair count");
    }
    const std::size_t pair_count =
        request.token_count * request.head_count * pairs_per_head;
    const std::size_t blocks = pair_count / kBlockSize +
                               (pair_count % kBlockSize != 0 ? 1 : 0);
    if (blocks > std::numeric_limits<unsigned int>::max()) {
        return CudaStatus::from(
            cudaErrorInvalidConfiguration, "CudaRopeBackend::apply grid");
    }

    rope_kernel<<<
        static_cast<unsigned int>(blocks),
        static_cast<unsigned int>(kBlockSize),
        0,
        stream.native_handle()>>>(
            request.values,
            pair_count,
            request.head_count,
            request.head_dimension,
            request.rotary_dimension,
            request.position_offset,
            request.frequency_base,
            request.pairing);
    return check_last_cuda_error("rope_kernel launch");
}

CudaStatus CudaRopeBackend::apply_batch(
    const core::RotaryEmbeddingBatchRequest& request,
    const CudaStream& stream) const {
    if (!request.validate().is_ok()) {
        return CudaStatus::from(
            cudaErrorInvalidValue, "CudaRopeBackend::apply_batch request");
    }
    if (!stream.valid()) {
        return CudaStatus::from(
            cudaErrorInvalidResourceHandle, "CudaRopeBackend::apply_batch stream");
    }
    const std::size_t pairs_per_head = request.rotary_dimension / 2;
    if (request.batch_size > std::numeric_limits<std::size_t>::max() /
                                 request.head_count ||
        request.batch_size * request.head_count >
            std::numeric_limits<std::size_t>::max() / pairs_per_head) {
        return CudaStatus::from(
            cudaErrorInvalidConfiguration, "CudaRopeBackend::apply_batch pair count");
    }
    const std::size_t pair_count =
        request.batch_size * request.head_count * pairs_per_head;
    const std::size_t blocks = pair_count / kBlockSize +
        (pair_count % kBlockSize != 0 ? 1 : 0);
    if (blocks > std::numeric_limits<unsigned int>::max()) {
        return CudaStatus::from(
            cudaErrorInvalidConfiguration, "CudaRopeBackend::apply_batch grid");
    }
    rope_batch_kernel<<<
        static_cast<unsigned int>(blocks),
        static_cast<unsigned int>(kBlockSize),
        0,
        stream.native_handle()>>>(
            request.values,
            request.positions,
            pair_count,
            request.head_count,
            request.head_dimension,
            request.rotary_dimension,
            request.frequency_base,
            request.pairing);
    return check_last_cuda_error("rope_batch_kernel launch");
}

const CudaRopeBackend& cuda_rope_backend() {
    static const CudaRopeBackend backend;
    return backend;
}

}  // namespace cpp_llm_engine::cuda
