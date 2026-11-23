// CUDA Backend: compute Router logits, global Softmax and stable Top-K on device.
#include "cpp_llm_engine/cuda/moe_router_backend.h"

#include "cpp_llm_engine/cuda/cuda_stream.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cfloat>
#include <limits>

namespace cpp_llm_engine::cuda {
namespace {

constexpr std::size_t kMaximumExperts = 256;

__global__ void moe_router_kernel(CudaMoeRouterRequest request) {
    __shared__ float logits[kMaximumExperts];
    const std::size_t token = blockIdx.x;
    const std::size_t expert = threadIdx.x;
    if (expert < request.expert_count) {
        const float* hidden = request.hidden + token * request.hidden_size;
        const float* weights = request.router + expert * request.hidden_size;
        float logit = 0.0F;
        for (std::size_t column = 0; column < request.hidden_size; ++column) {
            logit += hidden[column] * weights[column];
        }
        logits[expert] = logit;
    }
    __syncthreads();

    if (expert != 0) return;
    float maximum = -FLT_MAX;
    for (std::size_t index = 0; index < request.expert_count; ++index) {
        maximum = fmaxf(maximum, logits[index]);
    }
    float denominator = 0.0F;
    for (std::size_t index = 0; index < request.expert_count; ++index) {
        denominator += expf(logits[index] - maximum);
    }

    float selected_sum = 0.0F;
    const std::size_t output_offset = token * request.top_k;
    for (std::size_t slot = 0; slot < request.top_k; ++slot) {
        std::size_t selected = request.expert_count;
        float selected_logit = -FLT_MAX;
        for (std::size_t index = 0; index < request.expert_count; ++index) {
            bool already_selected = false;
            for (std::size_t previous = 0; previous < slot; ++previous) {
                already_selected = already_selected ||
                    static_cast<std::size_t>(
                        request.expert_indices[output_offset + previous]) == index;
            }
            if (!already_selected &&
                (logits[index] > selected_logit ||
                 (logits[index] == selected_logit && index < selected))) {
                selected = index;
                selected_logit = logits[index];
            }
        }
        const float probability = expf(selected_logit - maximum) / denominator;
        request.expert_indices[output_offset + slot] =
            static_cast<float>(selected);
        request.route_weights[output_offset + slot] = probability;
        selected_sum += probability;
    }
    const float normalization = request.normalize_weights
        ? request.weight_scale / selected_sum
        : request.weight_scale;
    for (std::size_t slot = 0; slot < request.top_k; ++slot) {
        request.route_weights[output_offset + slot] *= normalization;
    }
}

__global__ void moe_group_routes_kernel(CudaMoeGroupRoutesRequest request) {
    const std::size_t expert = blockIdx.x;
    if (threadIdx.x != 0) return;
    const std::size_t capacity = request.token_count * request.top_k;
    std::size_t count = 0;
    for (std::size_t token = 0; token < request.token_count; ++token) {
        for (std::size_t slot = 0; slot < request.top_k; ++slot) {
            const std::size_t route = token * request.top_k + slot;
            if (static_cast<std::size_t>(request.expert_indices[route]) !=
                expert) {
                continue;
            }
            const std::size_t output = expert * capacity + count;
            request.grouped_rows[output] = static_cast<float>(token);
            request.grouped_weights[output] = request.route_weights[route];
            ++count;
        }
    }
    request.expert_counts[expert] = static_cast<float>(count);
}

}  // namespace

CudaStatus launch_moe_router(
    const CudaMoeRouterRequest& request,
    const CudaStream& stream) {
    if (request.hidden == nullptr || request.router == nullptr ||
        request.expert_indices == nullptr || request.route_weights == nullptr ||
        request.token_count == 0 || request.hidden_size == 0 ||
        request.expert_count == 0 || request.expert_count > kMaximumExperts ||
        request.top_k == 0 || request.top_k > request.expert_count ||
        !std::isfinite(request.weight_scale) || request.weight_scale <= 0.0F ||
        !stream.valid() || request.token_count >
            std::numeric_limits<unsigned int>::max()) {
        return CudaStatus::from(cudaErrorInvalidValue, "MoE Router request");
    }
    moe_router_kernel<<<static_cast<unsigned int>(request.token_count),
        static_cast<unsigned int>(request.expert_count), 0,
        stream.native_handle()>>>(request);
    return check_last_cuda_error("moe_router_kernel launch");
}

CudaStatus launch_moe_group_routes(
    const CudaMoeGroupRoutesRequest& request,
    const CudaStream& stream) {
    if (request.expert_indices == nullptr || request.route_weights == nullptr ||
        request.grouped_rows == nullptr || request.grouped_weights == nullptr ||
        request.expert_counts == nullptr || request.token_count == 0 ||
        request.expert_count == 0 || request.expert_count > kMaximumExperts ||
        request.top_k == 0 || request.top_k > request.expert_count ||
        !stream.valid() || request.expert_count >
            std::numeric_limits<unsigned int>::max()) {
        return CudaStatus::from(
            cudaErrorInvalidValue, "MoE route grouping request");
    }
    moe_group_routes_kernel<<<static_cast<unsigned int>(request.expert_count),
        1, 0, stream.native_handle()>>>(request);
    return check_last_cuda_error("moe_group_routes_kernel launch");
}

}  // namespace cpp_llm_engine::cuda
