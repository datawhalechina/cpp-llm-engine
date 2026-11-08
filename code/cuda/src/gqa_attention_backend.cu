// CUDA Backend：实现三阶段 GQA golden 与无完整 score workspace 的融合在线基线。
// 契约：每个 query head 映射到 query_head / (Hq/Hkv) 的 KV head。
// 验证：prefill/decode 均与独立 CPU 公式对照，固定 workload 比较两条路径。
#include "cpp_llm_engine/cuda/gqa_attention_backend.h"

#include "cpp_llm_engine/cuda/cuda_stream.h"

#include <cuda_runtime.h>
#include <math_constants.h>

#include <cmath>
#include <limits>

namespace cpp_llm_engine::cuda {
namespace {

constexpr std::size_t kBlockSize = 256;
constexpr unsigned int kOnlineBlockSize = 128;
constexpr std::size_t kOnlineKeyTile = 32;

__global__ void fused_online_gqa_kernel(core::GqaAttentionRequest request, float scale) {
    const std::size_t row = blockIdx.x;
    const std::size_t query_head = row % request.query_head_count;
    const std::size_t query_token = row / request.query_head_count;
    const std::size_t group_size = request.query_head_count / request.kv_head_count;
    const std::size_t kv_head = query_head / group_size;
    const std::size_t query_position = request.query_position_offset + query_token;
    const std::size_t visible_keys = request.causal && query_position < request.key_tokens
                                         ? query_position + 1
                                         : request.key_tokens;
    const float* query_vector = request.query +
        (query_token * request.query_head_count + query_head) * request.head_dimension;

    __shared__ float tile_scores[kOnlineKeyTile];
    __shared__ float running_max;
    __shared__ float running_denominator;
    __shared__ float running_factor;
    __shared__ float tile_factor;
    if (threadIdx.x == 0) {
        running_max = -CUDART_INF_F;
        running_denominator = 0.0F;
    }
    __syncthreads();

    float accumulator = 0.0F;
    for (std::size_t tile_begin = 0; tile_begin < visible_keys;
         tile_begin += kOnlineKeyTile) {
        const std::size_t remaining = visible_keys - tile_begin;
        const std::size_t tile_count =
            remaining < kOnlineKeyTile ? remaining : kOnlineKeyTile;
        constexpr unsigned int warp_size = 32;
        constexpr unsigned int warps_per_block = kOnlineBlockSize / warp_size;
        const unsigned int warp = threadIdx.x / warp_size;
        const unsigned int lane = threadIdx.x % warp_size;
        for (std::size_t group_begin = 0; group_begin < tile_count;
             group_begin += warps_per_block) {
            const std::size_t tile_index = group_begin + warp;
            float partial = 0.0F;
            if (tile_index < tile_count) {
                const std::size_t key_token = tile_begin + tile_index;
                const float* key_vector = request.key +
                    (key_token * request.kv_head_count + kv_head) *
                        request.head_dimension;
                for (std::size_t dimension = lane;
                     dimension < request.head_dimension; dimension += warp_size) {
                    partial += query_vector[dimension] * key_vector[dimension];
                }
            }
            for (unsigned int offset = warp_size / 2; offset > 0; offset /= 2) {
                partial += __shfl_down_sync(0xFFFFFFFFU, partial, offset);
            }
            if (lane == 0 && tile_index < tile_count) {
                tile_scores[tile_index] = partial * scale;
            }
            __syncthreads();
        }

        if (threadIdx.x == 0) {
            float tile_max = -CUDART_INF_F;
            for (std::size_t index = 0; index < tile_count; ++index) {
                tile_max = fmaxf(tile_max, tile_scores[index]);
            }
            float tile_denominator = 0.0F;
            for (std::size_t index = 0; index < tile_count; ++index) {
                tile_scores[index] = expf(tile_scores[index] - tile_max);
                tile_denominator += tile_scores[index];
            }
            const float merged_max = fmaxf(running_max, tile_max);
            running_factor = running_denominator == 0.0F
                                 ? 0.0F
                                 : expf(running_max - merged_max);
            tile_factor = expf(tile_max - merged_max);
            running_denominator = running_factor * running_denominator +
                                  tile_factor * tile_denominator;
            running_max = merged_max;
        }
        __syncthreads();

        if (threadIdx.x < request.head_dimension) {
            float tile_output = 0.0F;
            for (std::size_t index = 0; index < tile_count; ++index) {
                const std::size_t value_index =
                    ((tile_begin + index) * request.kv_head_count + kv_head) *
                        request.head_dimension + threadIdx.x;
                tile_output += tile_scores[index] * request.value[value_index];
            }
            accumulator = running_factor * accumulator + tile_factor * tile_output;
        }
        __syncthreads();
    }

    if (threadIdx.x < request.head_dimension) {
        const std::size_t output_index =
            (query_token * request.query_head_count + query_head) *
                request.head_dimension + threadIdx.x;
        request.output[output_index] = accumulator / running_denominator;
    }
}

__global__ void gqa_score_kernel(
    const float* query,
    const float* key,
    float* scores,
    std::size_t score_count,
    std::size_t key_tokens,
    std::size_t query_head_count,
    std::size_t kv_head_count,
    std::size_t head_dimension,
    std::size_t query_position_offset,
    bool causal,
    float scale) {
    const std::size_t index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= score_count) return;
    const std::size_t key_token = index % key_tokens;
    const std::size_t query_head_token = index / key_tokens;
    const std::size_t query_head = query_head_token % query_head_count;
    const std::size_t query_token = query_head_token / query_head_count;
    if (causal && key_token > query_position_offset + query_token) {
        scores[index] = -CUDART_INF_F;
        return;
    }
    const std::size_t group_size = query_head_count / kv_head_count;
    const std::size_t kv_head = query_head / group_size;
    const float* query_vector = query +
        (query_token * query_head_count + query_head) * head_dimension;
    const float* key_vector = key +
        (key_token * kv_head_count + kv_head) * head_dimension;
    float score = 0.0F;
    for (std::size_t dimension = 0; dimension < head_dimension; ++dimension) {
        score += query_vector[dimension] * key_vector[dimension];
    }
    scores[index] = score * scale;
}

__global__ void stable_softmax_kernel(
    float* scores,
    std::size_t rows,
    std::size_t key_tokens) {
    const std::size_t row =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    float* values = scores + row * key_tokens;
    float maximum = -CUDART_INF_F;
    for (std::size_t key_token = 0; key_token < key_tokens; ++key_token) {
        maximum = fmaxf(maximum, values[key_token]);
    }
    float denominator = 0.0F;
    for (std::size_t key_token = 0; key_token < key_tokens; ++key_token) {
        const float probability = isinf(values[key_token]) && values[key_token] < 0.0F
                                      ? 0.0F
                                      : expf(values[key_token] - maximum);
        values[key_token] = probability;
        denominator += probability;
    }
    for (std::size_t key_token = 0; key_token < key_tokens; ++key_token) {
        values[key_token] /= denominator;
    }
}

__global__ void gqa_value_kernel(
    const float* probabilities,
    const float* value,
    float* output,
    std::size_t output_count,
    std::size_t key_tokens,
    std::size_t query_head_count,
    std::size_t kv_head_count,
    std::size_t head_dimension) {
    const std::size_t index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= output_count) return;
    const std::size_t dimension = index % head_dimension;
    const std::size_t query_head_token = index / head_dimension;
    const std::size_t query_head = query_head_token % query_head_count;
    const std::size_t group_size = query_head_count / kv_head_count;
    const std::size_t kv_head = query_head / group_size;
    const float* row = probabilities + query_head_token * key_tokens;
    float result = 0.0F;
    for (std::size_t key_token = 0; key_token < key_tokens; ++key_token) {
        const std::size_t value_index =
            (key_token * kv_head_count + kv_head) * head_dimension + dimension;
        result += row[key_token] * value[value_index];
    }
    output[index] = result;
}

__device__ std::size_t paged_kv_index(
    std::size_t logical_token,
    std::size_t kv_head,
    std::size_t dimension,
    const std::size_t* physical_blocks,
    std::size_t layer,
    std::size_t total_blocks,
    std::size_t block_size,
    std::size_t kv_head_count,
    std::size_t head_dimension) {
    const std::size_t logical_block = logical_token / block_size;
    const std::size_t slot = logical_token % block_size;
    const std::size_t physical_block = physical_blocks[logical_block];
    return ((((layer * total_blocks + physical_block) * block_size + slot) *
             kv_head_count + kv_head) * head_dimension + dimension);
}

__global__ void paged_gqa_score_kernel(
    CudaPagedGqaAttentionRequest request,
    float* scores,
    std::size_t score_count,
    float scale) {
    const std::size_t index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= score_count) return;
    const std::size_t key_token = index % request.key_tokens;
    const std::size_t query_head_token = index / request.key_tokens;
    const std::size_t query_head = query_head_token % request.query_head_count;
    const std::size_t query_token = query_head_token / request.query_head_count;
    if (request.causal && key_token > request.query_position_offset + query_token) {
        scores[index] = -CUDART_INF_F;
        return;
    }
    const std::size_t group_size = request.query_head_count / request.kv_head_count;
    const std::size_t kv_head = query_head / group_size;
    const float* query_vector = request.query +
        (query_token * request.query_head_count + query_head) * request.head_dimension;
    float score = 0.0F;
    for (std::size_t dimension = 0; dimension < request.head_dimension; ++dimension) {
        const std::size_t key_index = paged_kv_index(
            key_token, kv_head, dimension, request.physical_blocks, request.layer,
            request.total_blocks, request.block_size, request.kv_head_count,
            request.head_dimension);
        score += query_vector[dimension] * request.key[key_index];
    }
    scores[index] = score * scale;
}

__global__ void paged_gqa_value_kernel(
    CudaPagedGqaAttentionRequest request,
    const float* probabilities,
    std::size_t output_count) {
    const std::size_t index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= output_count) return;
    const std::size_t dimension = index % request.head_dimension;
    const std::size_t query_head_token = index / request.head_dimension;
    const std::size_t query_head = query_head_token % request.query_head_count;
    const std::size_t group_size = request.query_head_count / request.kv_head_count;
    const std::size_t kv_head = query_head / group_size;
    const float* row = probabilities + query_head_token * request.key_tokens;
    float result = 0.0F;
    for (std::size_t key_token = 0; key_token < request.key_tokens; ++key_token) {
        const std::size_t value_index = paged_kv_index(
            key_token, kv_head, dimension, request.physical_blocks, request.layer,
            request.total_blocks, request.block_size, request.kv_head_count,
            request.head_dimension);
        result += row[key_token] * request.value[value_index];
    }
    request.output[index] = result;
}

CudaStatus launch_status(const char* operation) {
    return check_last_cuda_error(operation);
}

}  // namespace

core::Status CudaPagedGqaAttentionRequest::validate() const {
    if (query == nullptr || key == nullptr || value == nullptr ||
        physical_blocks == nullptr || output == nullptr || query_tokens == 0 ||
        key_tokens == 0 || query_head_count == 0 || kv_head_count == 0 ||
        head_dimension == 0 || query_head_count % kv_head_count != 0 ||
        layer_count == 0 || layer >= layer_count || total_blocks == 0 ||
        block_size == 0 || physical_block_count == 0 ||
        physical_block_count <
            key_tokens / block_size + (key_tokens % block_size != 0 ? 1 : 0)) {
        return core::Status::error("CUDA paged GQA request is invalid");
    }
    if (query_tokens > std::numeric_limits<std::size_t>::max() / query_head_count ||
        query_tokens * query_head_count >
            std::numeric_limits<std::size_t>::max() / key_tokens ||
        query_tokens * query_head_count >
            std::numeric_limits<std::size_t>::max() / head_dimension) {
        return core::Status::error("CUDA paged GQA element count overflows");
    }
    return core::Status::ok();
}

std::size_t CudaPagedGqaAttentionRequest::score_count() const {
    return query_tokens * query_head_count * key_tokens;
}

core::BackendKind CudaGqaAttentionBackend::kind() const {
    return core::BackendKind::CUDA;
}

std::string_view CudaGqaAttentionBackend::name() const {
    return "cuda-gqa-attention";
}

CudaStatus CudaGqaAttentionBackend::compute(
    const core::GqaAttentionRequest& request,
    float* score_workspace,
    std::size_t workspace_elements,
    const CudaStream& stream) const {
    if (!request.validate().is_ok() || score_workspace == nullptr ||
        workspace_elements < request.score_count()) {
        return CudaStatus::from(cudaErrorInvalidValue, "CUDA GQA request/workspace");
    }
    if (!stream.valid()) {
        return CudaStatus::from(cudaErrorInvalidResourceHandle, "CUDA GQA stream");
    }
    const std::size_t score_count = request.score_count();
    const std::size_t row_count = request.query_tokens * request.query_head_count;
    const std::size_t output_count = row_count * request.head_dimension;
    const auto blocks_for = [](std::size_t count) {
        return count / kBlockSize + (count % kBlockSize != 0 ? 1 : 0);
    };
    const std::size_t score_blocks = blocks_for(score_count);
    const std::size_t row_blocks = blocks_for(row_count);
    const std::size_t output_blocks = blocks_for(output_count);
    if (score_blocks > std::numeric_limits<unsigned int>::max() ||
        row_blocks > std::numeric_limits<unsigned int>::max() ||
        output_blocks > std::numeric_limits<unsigned int>::max()) {
        return CudaStatus::from(cudaErrorInvalidConfiguration, "CUDA GQA grid");
    }
    const float scale = request.scale == 0.0F
                            ? 1.0F / sqrtf(static_cast<float>(request.head_dimension))
                            : request.scale;
    gqa_score_kernel<<<static_cast<unsigned int>(score_blocks), kBlockSize, 0,
                       stream.native_handle()>>>(
        request.query, request.key, score_workspace, score_count,
        request.key_tokens, request.query_head_count, request.kv_head_count,
        request.head_dimension, request.query_position_offset, request.causal, scale);
    CudaStatus status = launch_status("gqa_score_kernel launch");
    if (!status.is_ok()) return status;
    stable_softmax_kernel<<<static_cast<unsigned int>(row_blocks), kBlockSize, 0,
                            stream.native_handle()>>>(
        score_workspace, row_count, request.key_tokens);
    status = launch_status("stable_softmax_kernel launch");
    if (!status.is_ok()) return status;
    gqa_value_kernel<<<static_cast<unsigned int>(output_blocks), kBlockSize, 0,
                       stream.native_handle()>>>(
        score_workspace, request.value, request.output, output_count,
        request.key_tokens, request.query_head_count, request.kv_head_count,
        request.head_dimension);
    return launch_status("gqa_value_kernel launch");
}

CudaStatus CudaGqaAttentionBackend::compute_online(
    const core::GqaAttentionRequest& request,
    const CudaStream& stream) const {
    if (!request.validate().is_ok() || request.head_dimension > kOnlineBlockSize) {
        return CudaStatus::from(cudaErrorInvalidValue, "CUDA online GQA request");
    }
    if (!stream.valid()) {
        return CudaStatus::from(cudaErrorInvalidResourceHandle, "CUDA online GQA stream");
    }
    const std::size_t rows = request.query_tokens * request.query_head_count;
    if (rows > std::numeric_limits<unsigned int>::max()) {
        return CudaStatus::from(cudaErrorInvalidConfiguration, "CUDA online GQA grid");
    }
    const float scale = request.scale == 0.0F
                            ? 1.0F / sqrtf(static_cast<float>(request.head_dimension))
                            : request.scale;
    fused_online_gqa_kernel<<<static_cast<unsigned int>(rows), kOnlineBlockSize, 0,
                              stream.native_handle()>>>(request, scale);
    return launch_status("fused_online_gqa_kernel launch");
}

CudaStatus CudaGqaAttentionBackend::compute_paged(
    const CudaPagedGqaAttentionRequest& request,
    float* score_workspace,
    std::size_t workspace_elements,
    const CudaStream& stream) const {
    if (!request.validate().is_ok() || score_workspace == nullptr ||
        workspace_elements < request.score_count()) {
        return CudaStatus::from(
            cudaErrorInvalidValue, "CUDA paged GQA request/workspace");
    }
    if (!stream.valid()) {
        return CudaStatus::from(cudaErrorInvalidResourceHandle, "CUDA paged GQA stream");
    }
    const std::size_t score_count = request.score_count();
    const std::size_t row_count = request.query_tokens * request.query_head_count;
    const std::size_t output_count = row_count * request.head_dimension;
    const auto blocks_for = [](std::size_t count) {
        return count / kBlockSize + (count % kBlockSize != 0 ? 1 : 0);
    };
    const std::size_t score_blocks = blocks_for(score_count);
    const std::size_t row_blocks = blocks_for(row_count);
    const std::size_t output_blocks = blocks_for(output_count);
    if (score_blocks > std::numeric_limits<unsigned int>::max() ||
        row_blocks > std::numeric_limits<unsigned int>::max() ||
        output_blocks > std::numeric_limits<unsigned int>::max()) {
        return CudaStatus::from(cudaErrorInvalidConfiguration, "CUDA paged GQA grid");
    }
    const float scale = request.scale == 0.0F
        ? 1.0F / sqrtf(static_cast<float>(request.head_dimension))
        : request.scale;
    paged_gqa_score_kernel<<<static_cast<unsigned int>(score_blocks), kBlockSize, 0,
                              stream.native_handle()>>>(
        request, score_workspace, score_count, scale);
    CudaStatus status = launch_status("paged_gqa_score_kernel launch");
    if (!status.is_ok()) return status;
    stable_softmax_kernel<<<static_cast<unsigned int>(row_blocks), kBlockSize, 0,
                            stream.native_handle()>>>(
        score_workspace, row_count, request.key_tokens);
    status = launch_status("paged stable_softmax_kernel launch");
    if (!status.is_ok()) return status;
    paged_gqa_value_kernel<<<static_cast<unsigned int>(output_blocks), kBlockSize, 0,
                              stream.native_handle()>>>(
        request, score_workspace, output_count);
    return launch_status("paged_gqa_value_kernel launch");
}

const CudaGqaAttentionBackend& cuda_gqa_attention_backend() {
    static const CudaGqaAttentionBackend backend;
    return backend;
}

}  // namespace cpp_llm_engine::cuda
