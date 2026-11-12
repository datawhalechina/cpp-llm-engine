// CUDA Backend：分阶段实现压缩 MLA，避免为每个 value 重算注意力分数。
#include "cpp_llm_engine/cuda/mla_attention_backend.h"

#include "cpp_llm_engine/cuda/cuda_mla_cache.h"
#include "cpp_llm_engine/cuda/cuda_stream.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <limits>

namespace cpp_llm_engine::cuda {
namespace {

bool checked_add(
    std::size_t left, std::size_t right, std::size_t* output) {
    if (output == nullptr ||
        left > std::numeric_limits<std::size_t>::max() - right) {
        return false;
    }
    *output = left + right;
    return true;
}

bool checked_multiply(
    std::size_t left, std::size_t right, std::size_t* output) {
    if (output == nullptr || (left != 0 &&
        right > std::numeric_limits<std::size_t>::max() / left)) {
        return false;
    }
    *output = left * right;
    return true;
}

__global__ void mla_prepare_kernel(CudaMlaPrepareRequest request) {
    const std::size_t index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t head_width =
        request.nope_dimension + request.rope_dimension;
    const std::size_t query_width = request.head_count * head_width;
    if (index < query_width) {
        const std::size_t head = index / head_width;
        const std::size_t local = index % head_width;
        if (local < request.nope_dimension) {
            request.query_nope[head * request.nope_dimension + local] =
                request.query[index];
        } else {
            request.query_rope[head * request.rope_dimension +
                local - request.nope_dimension] = request.query[index];
        }
    }
    if (index < request.latent_dimension) {
        request.latent[index] = request.compressed_kv[index];
    }
    if (index < request.rope_dimension) {
        request.key_rope[index] =
            request.compressed_kv[request.latent_dimension + index];
    }
}

__device__ float latent_projection(
    const float* matrix_row, const float* latent, std::size_t width) {
    float result = 0.0F;
    for (std::size_t index = 0; index < width; ++index) {
        result += matrix_row[index] * latent[index];
    }
    return result;
}

__device__ float score_for_token(
    const CudaMlaAttentionRequest request,
    const float* latent_cache,
    const float* rope_cache,
    std::size_t head,
    std::size_t token) {
    const float* latent = latent_cache + token * request.latent_dimension;
    const float* key_rope = rope_cache + token * request.rope_dimension;
    const float* query_nope = request.query_nope + head * request.nope_dimension;
    const float* query_rope = request.query_rope + head * request.rope_dimension;
    const std::size_t matrix_head_offset =
        head * (request.nope_dimension + request.value_dimension) *
        request.latent_dimension;
    float score = 0.0F;
    for (std::size_t index = 0; index < request.rope_dimension; ++index) {
        score += query_rope[index] * key_rope[index];
    }
    for (std::size_t index = 0; index < request.nope_dimension; ++index) {
        score += query_nope[index] * latent_projection(
            request.kv_b + matrix_head_offset +
                index * request.latent_dimension,
            latent, request.latent_dimension);
    }
    return score * request.scale;
}

__global__ void mla_score_kernel(
    CudaMlaAttentionRequest request,
    const float* latent_cache,
    const float* rope_cache,
    std::size_t token_count) {
    const std::size_t index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t count = request.head_count * token_count;
    if (index >= count) return;
    const std::size_t head = index / token_count;
    const std::size_t token = index % token_count;
    request.probabilities[head * request.probability_capacity_tokens + token] =
        score_for_token(request, latent_cache, rope_cache, head, token);
}

__global__ void mla_softmax_kernel(
    CudaMlaAttentionRequest request, std::size_t token_count) {
    const std::size_t head =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (head >= request.head_count) return;
    float* probabilities = request.probabilities +
        head * request.probability_capacity_tokens;
    float maximum = -3.402823466e+38F;
    for (std::size_t token = 0; token < token_count; ++token) {
        maximum = fmaxf(maximum, probabilities[token]);
    }
    float denominator = 0.0F;
    for (std::size_t token = 0; token < token_count; ++token) {
        probabilities[token] = expf(probabilities[token] - maximum);
        denominator += probabilities[token];
    }
    for (std::size_t token = 0; token < token_count; ++token) {
        probabilities[token] /= denominator;
    }
}

__global__ void mla_value_kernel(
    CudaMlaAttentionRequest request,
    const float* latent_cache,
    std::size_t token_count) {
    const std::size_t index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t count = request.head_count * request.value_dimension;
    if (index >= count) return;
    const std::size_t head = index / request.value_dimension;
    const std::size_t value = index % request.value_dimension;
    const std::size_t matrix_head_offset =
        head * (request.nope_dimension + request.value_dimension) *
        request.latent_dimension;
    const float* value_row = request.kv_b + matrix_head_offset +
        (request.nope_dimension + value) * request.latent_dimension;
    const float* probabilities = request.probabilities +
        head * request.probability_capacity_tokens;
    float result = 0.0F;
    for (std::size_t token = 0; token < token_count; ++token) {
        result += probabilities[token] * latent_projection(
            value_row, latent_cache + token * request.latent_dimension,
            request.latent_dimension);
    }
    request.output[index] = result;
}

}  // namespace

core::Status CudaMlaPrepareRequest::validate() const {
    if (query == nullptr || compressed_kv == nullptr || query_nope == nullptr ||
        query_rope == nullptr || latent == nullptr || key_rope == nullptr ||
        head_count == 0 || nope_dimension == 0 || rope_dimension == 0 ||
        latent_dimension == 0) {
        return core::Status::error("MLA prepare request is invalid");
    }
    return core::Status::ok();
}

core::Status CudaMlaAttentionRequest::validate() const {
    if (query_nope == nullptr || query_rope == nullptr || kv_b == nullptr ||
        probabilities == nullptr || output == nullptr || head_count == 0 ||
        probability_capacity_tokens == 0 || nope_dimension == 0 ||
        rope_dimension == 0 || value_dimension == 0 || latent_dimension == 0 ||
        !std::isfinite(scale) || scale <= 0.0F) {
        return core::Status::error("MLA attention request is invalid");
    }
    return core::Status::ok();
}

core::BackendKind CudaMlaAttentionBackend::kind() const {
    return core::BackendKind::CUDA;
}
std::string_view CudaMlaAttentionBackend::name() const { return "cuda-mla"; }

CudaStatus CudaMlaAttentionBackend::compute(
    const CudaMlaAttentionRequest& request,
    const CudaMlaCache& cache,
    const CudaStream& stream) const {
    if (!request.validate().is_ok() || !cache.valid() ||
        cache.size_tokens() == 0 ||
        cache.size_tokens() > request.probability_capacity_tokens ||
        cache.latent_dimension() != request.latent_dimension ||
        cache.rope_dimension() != request.rope_dimension) {
        return CudaStatus::from(cudaErrorInvalidValue, "MLA attention request/cache");
    }
    if (!stream.valid()) {
        return CudaStatus::from(cudaErrorInvalidResourceHandle, "MLA stream");
    }
    std::size_t matrix_rows = 0;
    std::size_t matrix_elements_per_head = 0;
    std::size_t matrix_elements = 0;
    std::size_t probability_elements = 0;
    std::size_t score_count = 0;
    std::size_t value_count = 0;
    if (!checked_add(request.nope_dimension, request.value_dimension,
            &matrix_rows) ||
        !checked_multiply(matrix_rows, request.latent_dimension,
            &matrix_elements_per_head) ||
        !checked_multiply(request.head_count, matrix_elements_per_head,
            &matrix_elements) ||
        !checked_multiply(request.head_count,
            request.probability_capacity_tokens, &probability_elements) ||
        !checked_multiply(request.head_count, cache.size_tokens(),
            &score_count) ||
        !checked_multiply(request.head_count, request.value_dimension,
            &value_count)) {
        return CudaStatus::from(cudaErrorInvalidConfiguration, "MLA dimensions");
    }
    (void)matrix_elements;
    (void)probability_elements;
    constexpr unsigned int kThreads = 128;
    const std::size_t score_blocks = (score_count + kThreads - 1) / kThreads;
    const std::size_t head_blocks =
        (request.head_count + kThreads - 1) / kThreads;
    const std::size_t value_blocks = (value_count + kThreads - 1) / kThreads;
    if (score_blocks > std::numeric_limits<unsigned int>::max() ||
        head_blocks > std::numeric_limits<unsigned int>::max() ||
        value_blocks > std::numeric_limits<unsigned int>::max()) {
        return CudaStatus::from(cudaErrorInvalidConfiguration, "MLA grid");
    }
    mla_score_kernel<<<static_cast<unsigned int>(score_blocks), kThreads, 0,
        stream.native_handle()>>>(
            request, cache.latent_data(), cache.key_rope_data(),
            cache.size_tokens());
    CudaStatus status = check_last_cuda_error("mla_score_kernel launch");
    if (!status.is_ok()) return status;
    mla_softmax_kernel<<<static_cast<unsigned int>(head_blocks), kThreads, 0,
        stream.native_handle()>>>(request, cache.size_tokens());
    status = check_last_cuda_error("mla_softmax_kernel launch");
    if (!status.is_ok()) return status;
    mla_value_kernel<<<static_cast<unsigned int>(value_blocks), kThreads, 0,
        stream.native_handle()>>>(
            request, cache.latent_data(), cache.size_tokens());
    return check_last_cuda_error("mla_value_kernel launch");
}

CudaStatus CudaMlaAttentionBackend::prepare(
    const CudaMlaPrepareRequest& request,
    const CudaStream& stream) const {
    if (!request.validate().is_ok() || !stream.valid()) {
        return CudaStatus::from(cudaErrorInvalidValue, "MLA prepare request/stream");
    }
    std::size_t head_width = 0;
    std::size_t query_elements = 0;
    if (!checked_add(request.nope_dimension, request.rope_dimension,
            &head_width) ||
        !checked_multiply(request.head_count, head_width, &query_elements)) {
        return CudaStatus::from(cudaErrorInvalidConfiguration, "MLA prepare dimensions");
    }
    constexpr unsigned int kThreads = 256;
    const std::size_t count = std::max(
        query_elements,
        std::max(request.latent_dimension, request.rope_dimension));
    const std::size_t blocks = (count + kThreads - 1) / kThreads;
    if (blocks > std::numeric_limits<unsigned int>::max()) {
        return CudaStatus::from(cudaErrorInvalidConfiguration, "MLA prepare grid");
    }
    mla_prepare_kernel<<<static_cast<unsigned int>(blocks), kThreads, 0,
        stream.native_handle()>>>(request);
    return check_last_cuda_error("mla_prepare_kernel launch");
}

const CudaMlaAttentionBackend& cuda_mla_attention_backend() {
    static const CudaMlaAttentionBackend backend;
    return backend;
}

}  // namespace cpp_llm_engine::cuda
