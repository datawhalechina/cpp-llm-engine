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

__device__ std::size_t minimum_size(
    std::size_t left, std::size_t right) {
    return left < right ? left : right;
}

__global__ void mla_prepare_kernel(CudaMlaPrepareRequest request) {
    const std::size_t index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t head_width =
        request.nope_dimension + request.rope_dimension;
    const std::size_t query_width = request.head_count * head_width;
    const std::size_t query_elements = request.token_count * query_width;
    if (index < query_elements) {
        const std::size_t token = index / query_width;
        const std::size_t query_index = index % query_width;
        const std::size_t head = query_index / head_width;
        const std::size_t local = query_index % head_width;
        if (local < request.nope_dimension) {
            request.query_nope[
                (token * request.head_count + head) * request.nope_dimension +
                local] =
                request.query[index];
        } else {
            request.query_rope[
                (token * request.head_count + head) * request.rope_dimension +
                local - request.nope_dimension] = request.query[index];
        }
    }
    const std::size_t compressed_width =
        request.latent_dimension + request.rope_dimension;
    const std::size_t latent_elements =
        request.token_count * request.latent_dimension;
    if (index < latent_elements) {
        const std::size_t token = index / request.latent_dimension;
        const std::size_t local = index % request.latent_dimension;
        request.latent[index] =
            request.compressed_kv[token * compressed_width + local];
    }
    const std::size_t rope_elements =
        request.token_count * request.rope_dimension;
    if (index < rope_elements) {
        const std::size_t token = index / request.rope_dimension;
        const std::size_t local = index % request.rope_dimension;
        request.key_rope[index] = request.compressed_kv[
            token * compressed_width + request.latent_dimension + local];
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
    std::size_t query,
    std::size_t head,
    std::size_t token) {
    const float* latent = latent_cache + token * request.latent_dimension;
    const float* key_rope = rope_cache + token * request.rope_dimension;
    const float* query_nope = request.query_nope +
        (query * request.head_count + head) * request.nope_dimension;
    const float* query_rope = request.query_rope +
        (query * request.head_count + head) * request.rope_dimension;
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
    const std::size_t count =
        request.query_count * request.head_count * token_count;
    if (index >= count) return;
    const std::size_t query_head = index / token_count;
    const std::size_t query = query_head / request.head_count;
    const std::size_t head = query_head % request.head_count;
    const std::size_t token = index % token_count;
    request.probabilities[
        query_head * request.probability_capacity_tokens + token] =
        score_for_token(request, latent_cache, rope_cache, query, head, token);
}

__global__ void mla_softmax_kernel(
    CudaMlaAttentionRequest request, std::size_t token_count) {
    const std::size_t query_head =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (query_head >= request.query_count * request.head_count) return;
    const std::size_t query = query_head / request.head_count;
    const std::size_t visible_tokens = minimum_size(
        token_count, request.query_position_offset + query + 1);
    float* probabilities = request.probabilities +
        query_head * request.probability_capacity_tokens;
    float maximum = -3.402823466e+38F;
    for (std::size_t token = 0; token < visible_tokens; ++token) {
        maximum = fmaxf(maximum, probabilities[token]);
    }
    float denominator = 0.0F;
    for (std::size_t token = 0; token < visible_tokens; ++token) {
        probabilities[token] = expf(probabilities[token] - maximum);
        denominator += probabilities[token];
    }
    for (std::size_t token = 0; token < visible_tokens; ++token) {
        probabilities[token] /= denominator;
    }
}

__global__ void mla_value_kernel(
    CudaMlaAttentionRequest request,
    const float* latent_cache,
    std::size_t token_count) {
    const std::size_t index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t count = request.query_count * request.head_count *
        request.value_dimension;
    if (index >= count) return;
    const std::size_t query_head = index / request.value_dimension;
    const std::size_t query = query_head / request.head_count;
    const std::size_t head = query_head % request.head_count;
    const std::size_t value = index % request.value_dimension;
    const std::size_t matrix_head_offset =
        head * (request.nope_dimension + request.value_dimension) *
        request.latent_dimension;
    const float* value_row = request.kv_b + matrix_head_offset +
        (request.nope_dimension + value) * request.latent_dimension;
    const float* probabilities = request.probabilities +
        query_head * request.probability_capacity_tokens;
    const std::size_t visible_tokens = minimum_size(
        token_count, request.query_position_offset + query + 1);
    float result = 0.0F;
    for (std::size_t token = 0; token < visible_tokens; ++token) {
        result += probabilities[token] * latent_projection(
            value_row, latent_cache + token * request.latent_dimension,
            request.latent_dimension);
    }
    request.output[
        (query_head * request.value_dimension) + value] = result;
}

__device__ std::size_t paged_mla_offset(
    const CudaPagedMlaAttentionRequest& request,
    std::size_t token,
    std::size_t width) {
    const std::size_t logical_block = token / request.block_size;
    const std::size_t slot = token % request.block_size;
    const std::size_t physical_block = request.physical_blocks[logical_block];
    return ((request.layer * request.total_blocks + physical_block) *
        request.block_size + slot) * width;
}

__device__ float paged_score_for_token(
    const CudaPagedMlaAttentionRequest& paged,
    std::size_t query,
    std::size_t head,
    std::size_t token) {
    const auto& request = paged.attention;
    const float* latent = paged.latent + paged_mla_offset(
        paged, token, request.latent_dimension);
    const float* key_rope = paged.key_rope + paged_mla_offset(
        paged, token, request.rope_dimension);
    const float* query_nope = request.query_nope +
        (query * request.head_count + head) * request.nope_dimension;
    const float* query_rope = request.query_rope +
        (query * request.head_count + head) * request.rope_dimension;
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
            latent,
            request.latent_dimension);
    }
    return score * request.scale;
}

__global__ void paged_mla_score_kernel(
    CudaPagedMlaAttentionRequest request) {
    const std::size_t index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t count = request.attention.query_count *
        request.attention.head_count * request.token_count;
    if (index >= count) return;
    const std::size_t query_head = index / request.token_count;
    const std::size_t query = query_head / request.attention.head_count;
    const std::size_t head = query_head % request.attention.head_count;
    const std::size_t token = index % request.token_count;
    request.attention.probabilities[
        query_head * request.attention.probability_capacity_tokens + token] =
        paged_score_for_token(request, query, head, token);
}

__global__ void paged_mla_value_kernel(
    CudaPagedMlaAttentionRequest paged) {
    const auto request = paged.attention;
    const std::size_t index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t count = request.query_count * request.head_count *
        request.value_dimension;
    if (index >= count) return;
    const std::size_t query_head = index / request.value_dimension;
    const std::size_t query = query_head / request.head_count;
    const std::size_t head = query_head % request.head_count;
    const std::size_t value = index % request.value_dimension;
    const std::size_t matrix_head_offset =
        head * (request.nope_dimension + request.value_dimension) *
        request.latent_dimension;
    const float* value_row = request.kv_b + matrix_head_offset +
        (request.nope_dimension + value) * request.latent_dimension;
    const float* probabilities = request.probabilities +
        query_head * request.probability_capacity_tokens;
    const std::size_t visible_tokens = minimum_size(
        paged.token_count, request.query_position_offset + query + 1);
    float result = 0.0F;
    for (std::size_t token = 0; token < visible_tokens; ++token) {
        const float* latent = paged.latent + paged_mla_offset(
            paged, token, request.latent_dimension);
        result += probabilities[token] * latent_projection(
            value_row, latent, request.latent_dimension);
    }
    request.output[
        (query_head * request.value_dimension) + value] = result;
}

}  // namespace

core::Status CudaMlaPrepareRequest::validate() const {
    if (query == nullptr || compressed_kv == nullptr || query_nope == nullptr ||
        query_rope == nullptr || latent == nullptr || key_rope == nullptr ||
        head_count == 0 || nope_dimension == 0 || rope_dimension == 0 ||
        latent_dimension == 0 || token_count == 0) {
        return core::Status::error("MLA prepare request is invalid");
    }
    return core::Status::ok();
}

core::Status CudaMlaAttentionRequest::validate() const {
    if (query_nope == nullptr || query_rope == nullptr || kv_b == nullptr ||
        probabilities == nullptr || output == nullptr || head_count == 0 ||
        probability_capacity_tokens == 0 || nope_dimension == 0 ||
        rope_dimension == 0 || value_dimension == 0 || latent_dimension == 0 ||
        query_count == 0 || !std::isfinite(scale) || scale <= 0.0F) {
        return core::Status::error("MLA attention request is invalid");
    }
    return core::Status::ok();
}

core::Status CudaPagedMlaAttentionRequest::validate() const {
    const core::Status attention_status = attention.validate();
    if (!attention_status.is_ok()) return attention_status;
    if (latent == nullptr || key_rope == nullptr || physical_blocks == nullptr ||
        token_count == 0 || layer_count == 0 || layer >= layer_count ||
        total_blocks == 0 || block_size == 0 || physical_block_count == 0 ||
        token_count > attention.probability_capacity_tokens) {
        return core::Status::error("paged MLA request is invalid");
    }
    const std::size_t required_blocks = token_count / block_size +
        (token_count % block_size == 0 ? 0 : 1);
    if (physical_block_count < required_blocks) {
        return core::Status::error("paged MLA page table is too short");
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
        request.query_position_offset >= cache.size_tokens() ||
        request.query_count >
            cache.size_tokens() - request.query_position_offset ||
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
    std::size_t query_heads = 0;
    std::size_t score_count = 0;
    std::size_t value_count = 0;
    if (!checked_add(request.nope_dimension, request.value_dimension,
            &matrix_rows) ||
        !checked_multiply(matrix_rows, request.latent_dimension,
            &matrix_elements_per_head) ||
        !checked_multiply(request.head_count, matrix_elements_per_head,
            &matrix_elements) ||
        !checked_multiply(request.query_count, request.head_count,
            &query_heads) ||
        !checked_multiply(query_heads,
            request.probability_capacity_tokens, &probability_elements) ||
        !checked_multiply(query_heads, cache.size_tokens(),
            &score_count) ||
        !checked_multiply(query_heads, request.value_dimension,
            &value_count)) {
        return CudaStatus::from(cudaErrorInvalidConfiguration, "MLA dimensions");
    }
    (void)matrix_elements;
    (void)probability_elements;
    constexpr unsigned int kThreads = 128;
    const std::size_t score_blocks = (score_count + kThreads - 1) / kThreads;
    const std::size_t head_blocks =
        (query_heads + kThreads - 1) / kThreads;
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

CudaStatus CudaMlaAttentionBackend::compute_paged(
    const CudaPagedMlaAttentionRequest& request,
    const CudaStream& stream) const {
    if (!request.validate().is_ok()) {
        return CudaStatus::from(cudaErrorInvalidValue, "paged MLA request");
    }
    if (request.attention.query_position_offset >= request.token_count ||
        request.attention.query_count > request.token_count -
            request.attention.query_position_offset) {
        return CudaStatus::from(
            cudaErrorInvalidValue, "paged MLA query range");
    }
    if (!stream.valid()) {
        return CudaStatus::from(cudaErrorInvalidResourceHandle, "paged MLA stream");
    }
    std::size_t score_count = 0;
    std::size_t value_count = 0;
    std::size_t query_heads = 0;
    if (!checked_multiply(request.attention.query_count,
            request.attention.head_count, &query_heads) ||
        !checked_multiply(query_heads,
            request.token_count, &score_count) ||
        !checked_multiply(query_heads,
            request.attention.value_dimension,
            &value_count)) {
        return CudaStatus::from(
            cudaErrorInvalidConfiguration, "paged MLA dimensions");
    }
    constexpr unsigned int kThreads = 128;
    const std::size_t score_blocks = (score_count + kThreads - 1) / kThreads;
    const std::size_t head_blocks =
        (query_heads + kThreads - 1) / kThreads;
    const std::size_t value_blocks = (value_count + kThreads - 1) / kThreads;
    if (score_blocks > std::numeric_limits<unsigned int>::max() ||
        head_blocks > std::numeric_limits<unsigned int>::max() ||
        value_blocks > std::numeric_limits<unsigned int>::max()) {
        return CudaStatus::from(cudaErrorInvalidConfiguration, "paged MLA grid");
    }
    paged_mla_score_kernel<<<static_cast<unsigned int>(score_blocks),
        kThreads, 0, stream.native_handle()>>>(request);
    CudaStatus status = check_last_cuda_error("paged_mla_score_kernel launch");
    if (!status.is_ok()) return status;
    mla_softmax_kernel<<<static_cast<unsigned int>(head_blocks),
        kThreads, 0, stream.native_handle()>>>(
            request.attention, request.token_count);
    status = check_last_cuda_error("paged_mla_softmax_kernel launch");
    if (!status.is_ok()) return status;
    paged_mla_value_kernel<<<static_cast<unsigned int>(value_blocks),
        kThreads, 0, stream.native_handle()>>>(request);
    return check_last_cuda_error("paged_mla_value_kernel launch");
}

CudaStatus CudaMlaAttentionBackend::prepare(
    const CudaMlaPrepareRequest& request,
    const CudaStream& stream) const {
    if (!request.validate().is_ok() || !stream.valid()) {
        return CudaStatus::from(cudaErrorInvalidValue, "MLA prepare request/stream");
    }
    std::size_t head_width = 0;
    std::size_t query_elements = 0;
    std::size_t latent_elements = 0;
    std::size_t rope_elements = 0;
    if (!checked_add(request.nope_dimension, request.rope_dimension,
            &head_width) ||
        !checked_multiply(request.head_count, head_width, &query_elements) ||
        !checked_multiply(
            request.token_count, query_elements, &query_elements) ||
        !checked_multiply(request.token_count, request.latent_dimension,
            &latent_elements) ||
        !checked_multiply(request.token_count, request.rope_dimension,
            &rope_elements)) {
        return CudaStatus::from(cudaErrorInvalidConfiguration, "MLA prepare dimensions");
    }
    constexpr unsigned int kThreads = 256;
    const std::size_t count = std::max(
        query_elements,
        std::max(latent_elements, rope_elements));
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
