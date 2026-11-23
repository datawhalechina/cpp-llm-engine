// CUDA Backend：实现连续二维 Tensor 的索引行收集与加权散回。
#include "cpp_llm_engine/cuda/row_dispatch_backend.h"

#include "cpp_llm_engine/cuda/cuda_stream.h"

#include <cuda_runtime.h>

#include <limits>

namespace cpp_llm_engine::cuda {
namespace {

constexpr std::size_t kBlockSize = 256;

__global__ void gather_rows_kernel(CudaGatherRowsRequest request) {
    const std::size_t index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t elements = request.selected_rows * request.columns;
    if (index >= elements) return;
    const std::size_t selected_row = index / request.columns;
    const std::size_t column = index % request.columns;
    const std::size_t source_row = request.row_indices != nullptr
        ? request.row_indices[selected_row]
        : static_cast<std::size_t>(request.row_indices_f32[selected_row]);
    request.output[index] =
        request.source[source_row * request.columns + column];
}

__global__ void weighted_scatter_add_rows_kernel(
    CudaWeightedScatterAddRowsRequest request) {
    const std::size_t index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t elements = request.selected_rows * request.columns;
    if (index >= elements) return;
    const std::size_t selected_row = index / request.columns;
    const std::size_t column = index % request.columns;
    const std::size_t destination_row = request.destination_rows != nullptr
        ? request.destination_rows[selected_row]
        : static_cast<std::size_t>(
            request.destination_rows_f32[selected_row]);
    const std::size_t destination = destination_row * request.columns + column;
    request.output[destination] +=
        request.weights[selected_row] * request.source[index];
}

__global__ void spatial_merge_kernel(CudaSpatialMergeRequest request) {
    const std::size_t index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t output_grid_width =
        request.grid_width / request.merge_size;
    const std::size_t output_grid_height =
        request.grid_height / request.merge_size;
    const std::size_t merged_width = request.merge_size * request.merge_size *
        request.hidden_size;
    const std::size_t elements =
        output_grid_width * output_grid_height * merged_width;
    if (index >= elements) return;

    const std::size_t output_token = index / merged_width;
    const std::size_t merged_column = index % merged_width;
    const std::size_t hidden = merged_column % request.hidden_size;
    const std::size_t spatial = merged_column / request.hidden_size;
    const std::size_t dy = spatial / request.merge_size;
    const std::size_t dx = spatial % request.merge_size;
    const std::size_t output_y = output_token / output_grid_width;
    const std::size_t output_x = output_token % output_grid_width;
    const std::size_t input_y = output_y * request.merge_size + dy;
    const std::size_t input_x = output_x * request.merge_size + dx;
    const std::size_t input_token = input_y * request.grid_width + input_x;
    request.output[index] =
        request.input[input_token * request.hidden_size + hidden];
}

CudaStatus validate_grid(std::size_t elements, const char* operation) {
    const std::size_t blocks = elements / kBlockSize +
        (elements % kBlockSize != 0 ? 1 : 0);
    if (blocks > std::numeric_limits<unsigned int>::max()) {
        return CudaStatus::from(cudaErrorInvalidConfiguration, operation);
    }
    return CudaStatus::success();
}

}  // namespace

CudaStatus launch_gather_rows(
    const CudaGatherRowsRequest& request,
    const CudaStream& stream) {
    if (request.selected_rows == 0 || request.columns == 0) {
        return CudaStatus::success();
    }
    if (request.source == nullptr ||
        (request.row_indices == nullptr && request.row_indices_f32 == nullptr) ||
        (request.row_indices != nullptr && request.row_indices_f32 != nullptr) ||
        request.output == nullptr || request.source_rows == 0 ||
        !stream.valid()) {
        return CudaStatus::from(cudaErrorInvalidValue, "gather rows request");
    }
    const std::size_t elements = request.selected_rows * request.columns;
    CudaStatus status = validate_grid(elements, "gather rows grid");
    if (!status.is_ok()) return status;
    const std::size_t blocks = elements / kBlockSize +
        (elements % kBlockSize != 0 ? 1 : 0);
    gather_rows_kernel<<<static_cast<unsigned int>(blocks), kBlockSize, 0,
        stream.native_handle()>>>(request);
    return check_last_cuda_error("gather_rows_kernel launch");
}

CudaStatus launch_weighted_scatter_add_rows(
    const CudaWeightedScatterAddRowsRequest& request,
    const CudaStream& stream) {
    if (request.selected_rows == 0 || request.columns == 0) {
        return CudaStatus::success();
    }
    if (request.source == nullptr ||
        (request.destination_rows == nullptr &&
            request.destination_rows_f32 == nullptr) ||
        (request.destination_rows != nullptr &&
            request.destination_rows_f32 != nullptr) ||
        request.weights == nullptr || request.output == nullptr ||
        request.output_rows == 0 || !stream.valid()) {
        return CudaStatus::from(
            cudaErrorInvalidValue, "weighted scatter-add rows request");
    }
    const std::size_t elements = request.selected_rows * request.columns;
    CudaStatus status = validate_grid(elements, "weighted scatter-add rows grid");
    if (!status.is_ok()) return status;
    const std::size_t blocks = elements / kBlockSize +
        (elements % kBlockSize != 0 ? 1 : 0);
    weighted_scatter_add_rows_kernel<<<
        static_cast<unsigned int>(blocks), kBlockSize, 0,
        stream.native_handle()>>>(request);
    return check_last_cuda_error("weighted_scatter_add_rows_kernel launch");
}

CudaStatus launch_spatial_merge(
    const CudaSpatialMergeRequest& request,
    const CudaStream& stream) {
    if (request.input == nullptr || request.output == nullptr ||
        request.grid_width == 0 || request.grid_height == 0 ||
        request.merge_size == 0 || request.hidden_size == 0 ||
        request.grid_width % request.merge_size != 0 ||
        request.grid_height % request.merge_size != 0 || !stream.valid()) {
        return CudaStatus::from(cudaErrorInvalidValue, "spatial merge request");
    }
    const std::size_t output_grid_width =
        request.grid_width / request.merge_size;
    const std::size_t output_grid_height =
        request.grid_height / request.merge_size;
    const std::size_t elements = output_grid_width * output_grid_height *
        request.merge_size * request.merge_size * request.hidden_size;
    CudaStatus status = validate_grid(elements, "spatial merge grid");
    if (!status.is_ok()) return status;
    const std::size_t blocks = elements / kBlockSize +
        (elements % kBlockSize != 0 ? 1 : 0);
    spatial_merge_kernel<<<static_cast<unsigned int>(blocks), kBlockSize, 0,
        stream.native_handle()>>>(request);
    return check_last_cuda_error("spatial_merge_kernel launch");
}

}  // namespace cpp_llm_engine::cuda
