// CUDA Backend：声明连续二维 Tensor 的索引行收集与加权散回。
// 职责：用单次 kernel 替代逐行 Device-to-Device 小复制。
// 验证：ExecutionContext 组件测试覆盖乱序收集、加权累加和非法索引。
#pragma once

#include "cpp_llm_engine/cuda/cuda_status.h"

#include <cstddef>

namespace cpp_llm_engine::cuda {

class CudaStream;

struct CudaGatherRowsRequest final {
    const float* source = nullptr;
    const std::size_t* row_indices = nullptr;
    const float* row_indices_f32 = nullptr;
    float* output = nullptr;
    std::size_t source_rows = 0;
    std::size_t selected_rows = 0;
    std::size_t columns = 0;
};

struct CudaWeightedScatterAddRowsRequest final {
    const float* source = nullptr;
    const std::size_t* destination_rows = nullptr;
    const float* destination_rows_f32 = nullptr;
    const float* weights = nullptr;
    float* output = nullptr;
    std::size_t selected_rows = 0;
    std::size_t output_rows = 0;
    std::size_t columns = 0;
};

struct CudaSpatialMergeRequest final {
    const float* input = nullptr;
    float* output = nullptr;
    std::size_t grid_width = 0;
    std::size_t grid_height = 0;
    std::size_t merge_size = 0;
    std::size_t hidden_size = 0;
};

CudaStatus launch_gather_rows(
    const CudaGatherRowsRequest& request,
    const CudaStream& stream);
CudaStatus launch_weighted_scatter_add_rows(
    const CudaWeightedScatterAddRowsRequest& request,
    const CudaStream& stream);
CudaStatus launch_spatial_merge(
    const CudaSpatialMergeRequest& request,
    const CudaStream& stream);

}  // namespace cpp_llm_engine::cuda
