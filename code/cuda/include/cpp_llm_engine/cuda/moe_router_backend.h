// CUDA Backend: declare batched MoE Router and stable Top-K selection.
// The result stays compact: each token emits only top_k indices and weights.
#pragma once

#include "cpp_llm_engine/cuda/cuda_status.h"

#include <cstddef>

namespace cpp_llm_engine::cuda {

class CudaStream;

struct CudaMoeRouterRequest final {
    const float* hidden = nullptr;
    const float* router = nullptr;
    float* expert_indices = nullptr;
    float* route_weights = nullptr;
    std::size_t token_count = 0;
    std::size_t hidden_size = 0;
    std::size_t expert_count = 0;
    std::size_t top_k = 0;
    bool normalize_weights = false;
    float weight_scale = 1.0F;
};

struct CudaMoeGroupRoutesRequest final {
    const float* expert_indices = nullptr;
    const float* route_weights = nullptr;
    float* grouped_rows = nullptr;
    float* grouped_weights = nullptr;
    float* expert_counts = nullptr;
    std::size_t token_count = 0;
    std::size_t expert_count = 0;
    std::size_t top_k = 0;
};

CudaStatus launch_moe_router(
    const CudaMoeRouterRequest& request,
    const CudaStream& stream);
CudaStatus launch_moe_group_routes(
    const CudaMoeGroupRoutesRequest& request,
    const CudaStream& stream);

}  // namespace cpp_llm_engine::cuda
