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
    request.output[index] = request.source[
        request.row_indices[selected_row] * request.columns + column];
}

__global__ void weighted_scatter_add_rows_kernel(
    CudaWeightedScatterAddRowsRequest request) {
    const std::size_t index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t elements = request.selected_rows * request.columns;
    if (index >= elements) return;
    const std::size_t selected_row = index / request.columns;
    const std::size_t column = index % request.columns;
    const std::size_t destination =
        request.destination_rows[selected_row] * request.columns + column;
    request.output[destination] +=
        request.weights[selected_row] * request.source[index];
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
    if (request.source == nullptr || request.row_indices == nullptr ||
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
    if (request.source == nullptr || request.destination_rows == nullptr ||
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

}  // namespace cpp_llm_engine::cuda
