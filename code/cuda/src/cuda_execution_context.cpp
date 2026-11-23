// CUDA 执行层：实现 Tensor 级复制、算子分派和 workspace 管理。
// 契约：只接受连续 Float32 CUDA Tensor；所有工作进入上下文自己的 stream。
// 验证：cpp_llm_engine_cuda_execution_context_test。
#include "cpp_llm_engine/cuda/cuda_execution_context.h"

#include "cpp_llm_engine/core/elementwise_contract.h"
#include "cpp_llm_engine/core/decoder_elementwise_contract.h"
#include "cpp_llm_engine/core/gqa_attention_contract.h"
#include "cpp_llm_engine/core/matmul_contract.h"
#include "cpp_llm_engine/core/rope_contract.h"
#include "cpp_llm_engine/cuda/elementwise_backend.h"
#include "cpp_llm_engine/cuda/decoder_elementwise_backend.h"
#include "cpp_llm_engine/cuda/cuda_kv_cache.h"
#include "cpp_llm_engine/cuda/cuda_mla_cache.h"
#include "cpp_llm_engine/cuda/cuda_paged_mla_cache.h"
#include "cpp_llm_engine/cuda/cuda_paged_kv_cache.h"
#include "cpp_llm_engine/cuda/gqa_attention_backend.h"
#include "cpp_llm_engine/cuda/mla_attention_backend.h"
#include "cpp_llm_engine/cuda/moe_router_backend.h"
#include "cpp_llm_engine/cuda/quantized_dequantize_backend.h"
#include "cpp_llm_engine/cuda/quantized_gemv_backend.h"
#include "cpp_llm_engine/cuda/row_dispatch_backend.h"
#include "cpp_llm_engine/cuda/rope_backend.h"
#include "cpp_llm_engine/cuda/vision_elementwise_backend.h"

#include <cuda_runtime_api.h>

#include <limits>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace cpp_llm_engine::cuda {
namespace {

core::Status validate_tensor(const CudaTensor& tensor, const char* name) {
    const auto& contract = tensor.contract();
    if (tensor.empty() || contract.device != core::DeviceKind::CUDA ||
        contract.dtype != core::DType::Float32) {
        return core::Status::error(std::string(name) + " must be a Float32 CUDA Tensor");
    }
    return contract.validate_element_view(sizeof(float));
}

bool same_shape(const CudaTensor& lhs, const CudaTensor& rhs) {
    return lhs.contract().shape.dimensions == rhs.contract().shape.dimensions;
}

}  // namespace

core::Status CudaExecutionContext::create(CudaExecutionContext* output) {
    if (output == nullptr) return core::Status::error("CudaExecutionContext output is null");
    CudaExecutionContext replacement;
    const CudaStatus stream_status = CudaStream::create(&replacement.stream_);
    if (!stream_status.is_ok()) return stream_status.to_core_status();
    const CublasStatus matmul_status =
        CublasMatmulBackend::create(&replacement.matmul_backend_);
    if (!matmul_status.is_ok()) return matmul_status.to_core_status();
    *output = std::move(replacement);
    return core::Status::ok();
}

core::Status CudaExecutionContext::upload(
    CudaTensor* tensor,
    const void* source,
    std::size_t bytes) {
    if (!valid()) return core::Status::error("CudaExecutionContext is invalid");
    if (tensor == nullptr) return core::Status::error("upload tensor is null");
    const core::Status status = validate_tensor(*tensor, "upload tensor");
    if (!status.is_ok()) return status;
    return tensor->copy_from_host_async(source, bytes, stream_);
}

core::Status CudaExecutionContext::upload(
    CudaQuantizedMatrix* matrix,
    const void* source,
    std::size_t bytes) {
    if (!valid()) return core::Status::error("CudaExecutionContext is invalid");
    if (matrix == nullptr || matrix->empty()) {
        return core::Status::error("upload quantized matrix is null or empty");
    }
    return matrix->copy_from_host_async(source, bytes, stream_);
}

core::Status CudaExecutionContext::download(
    const CudaTensor& tensor,
    void* destination,
    std::size_t bytes) const {
    if (!valid()) return core::Status::error("CudaExecutionContext is invalid");
    const core::Status status = validate_tensor(tensor, "download tensor");
    if (!status.is_ok()) return status;
    return tensor.copy_to_host_async(destination, bytes, stream_);
}

core::Status CudaExecutionContext::copy(
    const CudaTensor& source,
    CudaTensor* destination) const {
    if (!valid()) return core::Status::error("CudaExecutionContext is invalid");
    if (destination == nullptr) return core::Status::error("copy destination is null");
    const core::Status source_status = validate_tensor(source, "copy source");
    if (!source_status.is_ok()) return source_status;
    const core::Status destination_status =
        validate_tensor(*destination, "copy destination");
    if (!destination_status.is_ok()) return destination_status;
    if (!same_shape(source, *destination)) {
        return core::Status::error("copy source and destination shapes must match");
    }
    if (source.data() == destination->data()) return core::Status::ok();
    return CudaStatus::from(
        cudaMemcpyAsync(
            destination->data(), source.data(), source.size_bytes(),
            cudaMemcpyDeviceToDevice, stream_.native_handle()),
        "CudaExecutionContext D2D copy").to_core_status();
}

core::Status CudaExecutionContext::pack_rows(
    const std::vector<const CudaTensor*>& rows,
    CudaTensor* batch) const {
    if (!valid()) return core::Status::error("CudaExecutionContext is invalid");
    if (rows.empty() || batch == nullptr) {
        return core::Status::error("pack rows requires inputs and batch output");
    }
    const core::Status batch_status = validate_tensor(*batch, "pack rows batch");
    if (!batch_status.is_ok()) return batch_status;
    const auto& batch_shape = batch->contract().shape.dimensions;
    if (batch_shape.size() < 2 || batch_shape[0] != rows.size()) {
        return core::Status::error("pack rows requires batch with leading B dimension");
    }
    std::size_t batch_elements = 0;
    if (!batch->contract().shape.checked_numel(&batch_elements).is_ok() ||
        batch_elements % rows.size() != 0) {
        return core::Status::error("pack rows batch element count is invalid");
    }
    const std::size_t row_elements = batch_elements / rows.size();
    for (const CudaTensor* row : rows) {
        if (row == nullptr) return core::Status::error("pack rows input is null");
        const core::Status row_status = validate_tensor(*row, "pack row");
        if (!row_status.is_ok()) return row_status;
        std::size_t elements = 0;
        if (!row->contract().shape.checked_numel(&elements).is_ok() ||
            elements != row_elements) {
            return core::Status::error("pack row elements do not match batch");
        }
    }
    const std::size_t row_bytes = row_elements * sizeof(float);
    auto* destination = static_cast<unsigned char*>(batch->data());
    for (std::size_t index = 0; index < rows.size(); ++index) {
        const CudaStatus status = CudaStatus::from(
            cudaMemcpyAsync(
                destination + index * row_bytes,
                rows[index]->data(),
                row_bytes,
                cudaMemcpyDeviceToDevice,
                stream_.native_handle()),
            "pack rows D2D copy");
        if (!status.is_ok()) return status.to_core_status();
    }
    return core::Status::ok();
}

core::Status CudaExecutionContext::unpack_rows(
    const CudaTensor& batch,
    const std::vector<CudaTensor*>& rows) const {
    if (!valid()) return core::Status::error("CudaExecutionContext is invalid");
    if (rows.empty()) return core::Status::error("unpack rows output is empty");
    const core::Status batch_status = validate_tensor(batch, "unpack rows batch");
    if (!batch_status.is_ok()) return batch_status;
    const auto& batch_shape = batch.contract().shape.dimensions;
    if (batch_shape.size() < 2 || batch_shape[0] != rows.size()) {
        return core::Status::error("unpack rows requires batch with leading B dimension");
    }
    std::size_t batch_elements = 0;
    if (!batch.contract().shape.checked_numel(&batch_elements).is_ok() ||
        batch_elements % rows.size() != 0) {
        return core::Status::error("unpack rows batch element count is invalid");
    }
    const std::size_t row_elements = batch_elements / rows.size();
    for (CudaTensor* row : rows) {
        if (row == nullptr) return core::Status::error("unpack row output is null");
        const core::Status row_status = validate_tensor(*row, "unpack row");
        if (!row_status.is_ok()) return row_status;
        std::size_t elements = 0;
        if (!row->contract().shape.checked_numel(&elements).is_ok() ||
            elements != row_elements) {
            return core::Status::error("unpack row elements do not match batch");
        }
    }
    const std::size_t row_bytes = row_elements * sizeof(float);
    const auto* source = static_cast<const unsigned char*>(batch.data());
    for (std::size_t index = 0; index < rows.size(); ++index) {
        const CudaStatus status = CudaStatus::from(
            cudaMemcpyAsync(
                rows[index]->data(),
                source + index * row_bytes,
                row_bytes,
                cudaMemcpyDeviceToDevice,
                stream_.native_handle()),
            "unpack rows D2D copy");
        if (!status.is_ok()) return status.to_core_status();
    }
    return core::Status::ok();
}

core::Status CudaExecutionContext::gather_rows(
    const CudaTensor& source,
    const std::vector<std::size_t>& row_indices,
    CudaTensor* output) {
    if (!valid() || output == nullptr || row_indices.empty()) {
        return core::Status::error("gather rows input is invalid");
    }
    const core::Status source_status = validate_tensor(source, "gather rows source");
    const core::Status output_status = validate_tensor(*output, "gather rows output");
    if (!source_status.is_ok()) return source_status;
    if (!output_status.is_ok()) return output_status;
    const auto& source_shape = source.contract().shape.dimensions;
    const auto& output_shape = output->contract().shape.dimensions;
    if (source_shape.size() != 2 || output_shape.size() != 2 ||
        output_shape[0] != row_indices.size() ||
        output_shape[1] != source_shape[1]) {
        return core::Status::error("gather rows shape mismatch");
    }
    for (const std::size_t row : row_indices) {
        if (row >= source_shape[0]) {
            return core::Status::error("gather row index is out of range");
        }
    }
    const std::size_t index_bytes = row_indices.size() * sizeof(std::size_t);
    core::Status status = reserve_workspace(index_bytes);
    if (!status.is_ok()) return status;
    CudaStatus cuda_status = CudaStatus::from(cudaMemcpyAsync(
        workspace_.data(), row_indices.data(), index_bytes,
        cudaMemcpyHostToDevice, stream_.native_handle()),
        "gather rows indices upload");
    if (!cuda_status.is_ok()) return cuda_status.to_core_status();
    const CudaGatherRowsRequest request{
        static_cast<const float*>(source.data()),
        static_cast<const std::size_t*>(workspace_.data()),
        nullptr,
        static_cast<float*>(output->data()), source_shape[0],
        row_indices.size(), source_shape[1]};
    return launch_gather_rows(request, stream_).to_core_status();
}

core::Status CudaExecutionContext::gather_rows_device(
    const CudaTensor& source,
    const CudaTensor& row_indices,
    std::size_t index_offset,
    std::size_t selected_rows,
    CudaTensor* output) const {
    if (!valid() || output == nullptr || selected_rows == 0) {
        return core::Status::error("device gather rows input is invalid");
    }
    const core::Status source_status =
        validate_tensor(source, "device gather rows source");
    const core::Status index_status =
        validate_tensor(row_indices, "device gather rows indices");
    const core::Status output_status =
        validate_tensor(*output, "device gather rows output");
    if (!source_status.is_ok()) return source_status;
    if (!index_status.is_ok()) return index_status;
    if (!output_status.is_ok()) return output_status;
    const auto& source_shape = source.contract().shape.dimensions;
    const auto& index_shape = row_indices.contract().shape.dimensions;
    const auto& output_shape = output->contract().shape.dimensions;
    std::size_t index_elements = 0;
    if (source_shape.size() != 2 || output_shape.size() != 2 ||
        !row_indices.contract().shape.checked_numel(&index_elements).is_ok() ||
        index_offset > index_elements ||
        selected_rows > index_elements - index_offset ||
        output_shape !=
            std::vector<std::size_t>{selected_rows, source_shape[1]}) {
        return core::Status::error("device gather rows shape mismatch");
    }
    const auto* indices = static_cast<const float*>(row_indices.data());
    const CudaGatherRowsRequest request{
        static_cast<const float*>(source.data()), nullptr,
        indices + index_offset, static_cast<float*>(output->data()),
        source_shape[0], selected_rows, source_shape[1]};
    return launch_gather_rows(request, stream_).to_core_status();
}

core::Status CudaExecutionContext::spatial_merge(
    const CudaTensor& input,
    std::size_t grid_width,
    std::size_t grid_height,
    std::size_t merge_size,
    CudaTensor* output) const {
    if (!valid() || output == nullptr || grid_width == 0 || grid_height == 0 ||
        merge_size == 0 || grid_width % merge_size != 0 ||
        grid_height % merge_size != 0) {
        return core::Status::error("spatial merge input is invalid");
    }
    const core::Status input_status = validate_tensor(input, "spatial merge input");
    const core::Status output_status = validate_tensor(*output, "spatial merge output");
    if (!input_status.is_ok()) return input_status;
    if (!output_status.is_ok()) return output_status;
    if (grid_width > std::numeric_limits<std::size_t>::max() / grid_height) {
        return core::Status::error("spatial merge grid overflows");
    }
    const auto& input_shape = input.contract().shape.dimensions;
    const auto& output_shape = output->contract().shape.dimensions;
    if (input_shape.size() != 2 || input_shape[0] != grid_width * grid_height) {
        return core::Status::error("spatial merge input shape mismatch");
    }
    if (merge_size > std::numeric_limits<std::size_t>::max() / merge_size ||
        merge_size * merge_size >
            std::numeric_limits<std::size_t>::max() / input_shape[1]) {
        return core::Status::error("spatial merge width overflows");
    }
    const std::size_t merged_width =
        merge_size * merge_size * input_shape[1];
    const std::vector<std::size_t> expected_output{
        (grid_width / merge_size) * (grid_height / merge_size), merged_width};
    if (output_shape != expected_output) {
        return core::Status::error("spatial merge output shape mismatch");
    }
    const CudaSpatialMergeRequest request{
        static_cast<const float*>(input.data()),
        static_cast<float*>(output->data()), grid_width, grid_height,
        merge_size, input_shape[1]};
    return launch_spatial_merge(request, stream_).to_core_status();
}

core::Status CudaExecutionContext::weighted_scatter_add_rows(
    const CudaTensor& source,
    const std::vector<std::size_t>& destination_rows,
    const std::vector<float>& weights,
    CudaTensor* output) {
    if (!valid() || output == nullptr || destination_rows.empty() ||
        destination_rows.size() != weights.size()) {
        return core::Status::error("weighted scatter-add rows input is invalid");
    }
    const core::Status source_status =
        validate_tensor(source, "weighted scatter-add rows source");
    const core::Status output_status =
        validate_tensor(*output, "weighted scatter-add rows output");
    if (!source_status.is_ok()) return source_status;
    if (!output_status.is_ok()) return output_status;
    const auto& source_shape = source.contract().shape.dimensions;
    const auto& output_shape = output->contract().shape.dimensions;
    if (source_shape.size() != 2 || output_shape.size() != 2 ||
        source_shape[0] != destination_rows.size() ||
        source_shape[1] != output_shape[1]) {
        return core::Status::error("weighted scatter-add rows shape mismatch");
    }
    for (const std::size_t row : destination_rows) {
        if (row >= output_shape[0]) {
            return core::Status::error(
                "weighted scatter-add destination is out of range");
        }
    }
    const std::unordered_set<std::size_t> unique_rows(
        destination_rows.begin(), destination_rows.end());
    if (unique_rows.size() != destination_rows.size()) {
        return core::Status::error(
            "weighted scatter-add destinations must be unique");
    }
    const std::size_t index_bytes = destination_rows.size() * sizeof(std::size_t);
    const std::size_t weight_bytes = weights.size() * sizeof(float);
    core::Status status = reserve_workspace(index_bytes + weight_bytes);
    if (!status.is_ok()) return status;
    auto* workspace = static_cast<unsigned char*>(workspace_.data());
    CudaStatus cuda_status = CudaStatus::from(cudaMemcpyAsync(
        workspace, destination_rows.data(), index_bytes,
        cudaMemcpyHostToDevice, stream_.native_handle()),
        "weighted scatter-add indices upload");
    if (cuda_status.is_ok()) cuda_status = CudaStatus::from(cudaMemcpyAsync(
        workspace + index_bytes, weights.data(), weight_bytes,
        cudaMemcpyHostToDevice, stream_.native_handle()),
        "weighted scatter-add weights upload");
    if (!cuda_status.is_ok()) return cuda_status.to_core_status();
    const CudaWeightedScatterAddRowsRequest request{
        static_cast<const float*>(source.data()),
        reinterpret_cast<const std::size_t*>(workspace),
        nullptr,
        reinterpret_cast<const float*>(workspace + index_bytes),
        static_cast<float*>(output->data()), destination_rows.size(),
        output_shape[0], output_shape[1]};
    return launch_weighted_scatter_add_rows(request, stream_).to_core_status();
}

core::Status CudaExecutionContext::weighted_scatter_add_rows_device(
    const CudaTensor& source,
    const CudaTensor& destination_rows,
    const CudaTensor& weights,
    std::size_t metadata_offset,
    CudaTensor* output) const {
    if (!valid() || output == nullptr) {
        return core::Status::error(
            "device weighted scatter-add input is invalid");
    }
    for (const auto& item : {
             std::pair<const CudaTensor*, const char*>{
                 &source, "device weighted scatter source"},
             {&destination_rows, "device weighted scatter destinations"},
             {&weights, "device weighted scatter weights"},
             {output, "device weighted scatter output"}}) {
        const core::Status status = validate_tensor(*item.first, item.second);
        if (!status.is_ok()) return status;
    }
    const auto& source_shape = source.contract().shape.dimensions;
    const auto& output_shape = output->contract().shape.dimensions;
    std::size_t destination_elements = 0;
    std::size_t weight_elements = 0;
    if (source_shape.size() != 2 || output_shape.size() != 2 ||
        source_shape[1] != output_shape[1] ||
        !destination_rows.contract().shape.checked_numel(
            &destination_elements).is_ok() ||
        !weights.contract().shape.checked_numel(&weight_elements).is_ok() ||
        destination_elements != weight_elements ||
        metadata_offset > destination_elements ||
        source_shape[0] > destination_elements - metadata_offset) {
        return core::Status::error(
            "device weighted scatter-add shape mismatch");
    }
    const auto* rows = static_cast<const float*>(destination_rows.data());
    const auto* route_weights = static_cast<const float*>(weights.data());
    const CudaWeightedScatterAddRowsRequest request{
        static_cast<const float*>(source.data()), nullptr,
        rows + metadata_offset, route_weights + metadata_offset,
        static_cast<float*>(output->data()), source_shape[0], output_shape[0],
        output_shape[1]};
    return launch_weighted_scatter_add_rows(request, stream_).to_core_status();
}

core::Status CudaExecutionContext::moe_route(
    const CudaTensor& hidden,
    const CudaTensor& router,
    std::size_t top_k,
    bool normalize_weights,
    float weight_scale,
    CudaTensor* expert_indices,
    CudaTensor* route_weights) const {
    if (!valid() || expert_indices == nullptr || route_weights == nullptr) {
        return core::Status::error("MoE Router context/output is invalid");
    }
    for (const auto& item : {
             std::pair<const CudaTensor*, const char*>{&hidden, "MoE hidden"},
             {&router, "MoE Router weights"},
             {expert_indices, "MoE expert indices"},
             {route_weights, "MoE route weights"}}) {
        const core::Status status = validate_tensor(*item.first, item.second);
        if (!status.is_ok()) return status;
    }
    const auto& hidden_shape = hidden.contract().shape.dimensions;
    const auto& router_shape = router.contract().shape.dimensions;
    const auto& index_shape = expert_indices->contract().shape.dimensions;
    if (hidden_shape.size() != 2 || router_shape.size() != 2 ||
        hidden_shape[1] != router_shape[1] || top_k == 0 ||
        index_shape != std::vector<std::size_t>{hidden_shape[0], top_k} ||
        route_weights->contract().shape.dimensions != index_shape) {
        return core::Status::error(
            "MoE Router requires hidden[B,D], router[E,D] and outputs[B,K]");
    }
    const CudaMoeRouterRequest request{
        static_cast<const float*>(hidden.data()),
        static_cast<const float*>(router.data()),
        static_cast<float*>(expert_indices->data()),
        static_cast<float*>(route_weights->data()),
        hidden_shape[0], hidden_shape[1], router_shape[0], top_k,
        normalize_weights, weight_scale};
    return launch_moe_router(request, stream_).to_core_status();
}

core::Status CudaExecutionContext::moe_group_routes(
    const CudaTensor& expert_indices,
    const CudaTensor& route_weights,
    std::size_t expert_count,
    CudaTensor* grouped_rows,
    CudaTensor* grouped_weights,
    CudaTensor* expert_counts) const {
    if (!valid() || grouped_rows == nullptr || grouped_weights == nullptr ||
        expert_counts == nullptr || expert_count == 0) {
        return core::Status::error("MoE route grouping input is invalid");
    }
    for (const auto& item : {
             std::pair<const CudaTensor*, const char*>{
                 &expert_indices, "MoE expert indices"},
             {&route_weights, "MoE route weights"},
             {grouped_rows, "MoE grouped rows"},
             {grouped_weights, "MoE grouped weights"},
             {expert_counts, "MoE expert counts"}}) {
        const core::Status status = validate_tensor(*item.first, item.second);
        if (!status.is_ok()) return status;
    }
    const auto& route_shape = expert_indices.contract().shape.dimensions;
    std::size_t route_elements = 0;
    if (route_shape.size() != 2 ||
        !expert_indices.contract().shape.checked_numel(&route_elements).is_ok() ||
        route_weights.contract().shape.dimensions != route_shape ||
        grouped_rows->contract().shape.dimensions !=
            std::vector<std::size_t>{expert_count, route_elements} ||
        grouped_weights->contract().shape.dimensions !=
            grouped_rows->contract().shape.dimensions ||
        expert_counts->contract().shape.dimensions !=
            std::vector<std::size_t>{expert_count}) {
        return core::Status::error("MoE route grouping shape mismatch");
    }
    const CudaMoeGroupRoutesRequest request{
        static_cast<const float*>(expert_indices.data()),
        static_cast<const float*>(route_weights.data()),
        static_cast<float*>(grouped_rows->data()),
        static_cast<float*>(grouped_weights->data()),
        static_cast<float*>(expert_counts->data()), route_shape[0],
        expert_count, route_shape[1]};
    return launch_moe_group_routes(request, stream_).to_core_status();
}

core::Status CudaExecutionContext::zero(CudaTensor* output) const {
    if (!valid() || output == nullptr) {
        return core::Status::error("zero output is invalid");
    }
    const core::Status status = validate_tensor(*output, "zero output");
    if (!status.is_ok()) return status;
    return CudaStatus::from(cudaMemsetAsync(
        output->data(), 0, output->size_bytes(), stream_.native_handle()),
        "zero CUDA Tensor").to_core_status();
}

core::Status CudaExecutionContext::vector_add(
    const CudaTensor& lhs,
    const CudaTensor& rhs,
    CudaTensor* output) const {
    if (!valid()) return core::Status::error("CudaExecutionContext is invalid");
    if (output == nullptr) return core::Status::error("vector_add output is null");
    for (const auto& item : {
             std::pair<const CudaTensor*, const char*>{&lhs, "vector_add lhs"},
             {&rhs, "vector_add rhs"},
             {output, "vector_add output"}}) {
        const core::Status status = validate_tensor(*item.first, item.second);
        if (!status.is_ok()) return status;
    }
    if (!same_shape(lhs, rhs) || !same_shape(lhs, *output)) {
        return core::Status::error("vector_add shapes must match");
    }
    std::size_t elements = 0;
    const core::Status count_status = lhs.contract().shape.checked_numel(&elements);
    if (!count_status.is_ok()) return count_status;
    const core::VectorAddRequest request{
        static_cast<const float*>(lhs.data()),
        static_cast<const float*>(rhs.data()),
        static_cast<float*>(output->data()),
        elements};
    return cuda_elementwise_backend().vector_add(request, stream_).to_core_status();
}

core::Status CudaExecutionContext::rms_norm(
    const CudaTensor& input,
    const CudaTensor& weight,
    CudaTensor* output,
    float epsilon) const {
    if (!valid() || output == nullptr) return core::Status::error("RMSNorm context/output invalid");
    if (!same_shape(input, weight) || !same_shape(input, *output)) {
        return core::Status::error("RMSNorm input, weight and output shapes must match");
    }
    std::size_t elements = 0;
    if (!input.contract().shape.checked_numel(&elements).is_ok()) {
        return core::Status::error("RMSNorm element count is invalid");
    }
    const core::RmsNormRequest request{
        static_cast<const float*>(input.data()),
        static_cast<const float*>(weight.data()),
        static_cast<float*>(output->data()), elements, epsilon};
    return cuda_decoder_elementwise_backend().rms_norm(request, stream_).to_core_status();
}

core::Status CudaExecutionContext::rms_norm_batch(
    const CudaTensor& input,
    const CudaTensor& weight,
    CudaTensor* output,
    float epsilon) const {
    if (!valid() || output == nullptr) {
        return core::Status::error("batch RMSNorm context/output invalid");
    }
    const core::Status input_status = validate_tensor(input, "batch RMSNorm input");
    if (!input_status.is_ok()) return input_status;
    const core::Status weight_status = validate_tensor(weight, "batch RMSNorm weight");
    if (!weight_status.is_ok()) return weight_status;
    const core::Status output_status = validate_tensor(*output, "batch RMSNorm output");
    if (!output_status.is_ok()) return output_status;
    const auto& input_shape = input.contract().shape.dimensions;
    const auto& weight_shape = weight.contract().shape.dimensions;
    if (input_shape.size() < 2 || weight_shape.size() != 1 ||
        output->contract().shape.dimensions != input_shape ||
        input_shape.back() != weight_shape[0]) {
        return core::Status::error(
            "batch RMSNorm requires input/output[...,D] and weight[D]");
    }
    std::size_t elements = 0;
    if (!input.contract().shape.checked_numel(&elements).is_ok() ||
        elements % input_shape.back() != 0) {
        return core::Status::error("batch RMSNorm element count is invalid");
    }
    const core::RmsNormBatchRequest request{
        static_cast<const float*>(input.data()),
        static_cast<const float*>(weight.data()),
        static_cast<float*>(output->data()),
        elements / input_shape.back(), input_shape.back(), epsilon};
    return cuda_decoder_elementwise_backend().rms_norm_batch(
        request, stream_).to_core_status();
}

core::Status CudaExecutionContext::layer_norm_batch(
    const CudaTensor& input,
    const CudaTensor& weight,
    const CudaTensor& bias,
    CudaTensor* output,
    float epsilon) const {
    if (!valid() || output == nullptr) {
        return core::Status::error("batch LayerNorm context/output invalid");
    }
    for (const auto& item : {
             std::pair<const CudaTensor*, const char*>{&input, "LayerNorm input"},
             {&weight, "LayerNorm weight"}, {&bias, "LayerNorm bias"},
             {output, "LayerNorm output"}}) {
        const core::Status status = validate_tensor(*item.first, item.second);
        if (!status.is_ok()) return status;
    }
    const auto& shape = input.contract().shape.dimensions;
    const auto& weight_shape = weight.contract().shape.dimensions;
    if (shape.size() < 2 || output->contract().shape.dimensions != shape ||
        weight_shape.size() != 1 ||
        bias.contract().shape.dimensions != weight_shape ||
        shape.back() != weight_shape[0]) {
        return core::Status::error(
            "batch LayerNorm requires input/output[...,D] and weight/bias[D]");
    }
    std::size_t elements = 0;
    if (!input.contract().shape.checked_numel(&elements).is_ok() ||
        elements % shape.back() != 0) {
        return core::Status::error("batch LayerNorm element count is invalid");
    }
    const CudaLayerNormBatchRequest request{
        static_cast<const float*>(input.data()),
        static_cast<const float*>(weight.data()),
        static_cast<const float*>(bias.data()),
        static_cast<float*>(output->data()), elements / shape.back(),
        shape.back(), epsilon};
    return launch_layer_norm_batch(request, stream_).to_core_status();
}

core::Status CudaExecutionContext::add_bias_batch(
    const CudaTensor& input,
    const CudaTensor& bias,
    CudaTensor* output) const {
    if (!valid() || output == nullptr) {
        return core::Status::error("batch bias context/output invalid");
    }
    const core::Status input_status = validate_tensor(input, "batch bias input");
    const core::Status bias_status = validate_tensor(bias, "batch bias values");
    const core::Status output_status = validate_tensor(*output, "batch bias output");
    if (!input_status.is_ok()) return input_status;
    if (!bias_status.is_ok()) return bias_status;
    if (!output_status.is_ok()) return output_status;
    const auto& shape = input.contract().shape.dimensions;
    const auto& bias_shape = bias.contract().shape.dimensions;
    std::size_t elements = 0;
    if (shape.empty() || bias_shape.size() != 1 || bias_shape[0] == 0 ||
        output->contract().shape.dimensions != shape ||
        !input.contract().shape.checked_numel(&elements).is_ok() ||
        elements % bias_shape[0] != 0) {
        return core::Status::error(
            "batch bias requires equal input/output shape and a trailing contiguous bias width");
    }
    const CudaBiasBatchRequest request{
        static_cast<const float*>(input.data()),
        static_cast<const float*>(bias.data()),
        static_cast<float*>(output->data()), elements, bias_shape[0]};
    return launch_bias_batch(request, stream_).to_core_status();
}

core::Status CudaExecutionContext::gelu(
    const CudaTensor& input,
    CudaTensor* output) const {
    if (!valid() || output == nullptr) {
        return core::Status::error("GELU context/output invalid");
    }
    const core::Status input_status = validate_tensor(input, "GELU input");
    const core::Status output_status = validate_tensor(*output, "GELU output");
    if (!input_status.is_ok()) return input_status;
    if (!output_status.is_ok()) return output_status;
    std::size_t elements = 0;
    if (!same_shape(input, *output) ||
        !input.contract().shape.checked_numel(&elements).is_ok()) {
        return core::Status::error("GELU input/output shape mismatch");
    }
    const CudaGeluRequest request{
        static_cast<const float*>(input.data()),
        static_cast<float*>(output->data()), elements};
    return launch_gelu(request, stream_).to_core_status();
}

core::Status CudaExecutionContext::swiglu(
    const CudaTensor& gate,
    const CudaTensor& up,
    CudaTensor* output) const {
    if (!valid() || output == nullptr) return core::Status::error("SwiGLU context/output invalid");
    if (!same_shape(gate, up) || !same_shape(gate, *output)) {
        return core::Status::error("SwiGLU gate, up and output shapes must match");
    }
    std::size_t elements = 0;
    if (!gate.contract().shape.checked_numel(&elements).is_ok()) {
        return core::Status::error("SwiGLU element count is invalid");
    }
    const core::SwiGluRequest request{
        static_cast<const float*>(gate.data()), static_cast<const float*>(up.data()),
        static_cast<float*>(output->data()), elements};
    return cuda_decoder_elementwise_backend().swiglu(request, stream_).to_core_status();
}

core::Status CudaExecutionContext::matmul(
    const CudaTensor& lhs,
    const CudaTensor& rhs,
    CudaTensor* output) const {
    if (!valid()) return core::Status::error("CudaExecutionContext is invalid");
    if (output == nullptr) return core::Status::error("matmul output is null");
    for (const auto& item : {
             std::pair<const CudaTensor*, const char*>{&lhs, "matmul lhs"},
             {&rhs, "matmul rhs"},
             {output, "matmul output"}}) {
        const core::Status status = validate_tensor(*item.first, item.second);
        if (!status.is_ok()) return status;
    }
    const auto& lhs_shape = lhs.contract().shape.dimensions;
    const auto& rhs_shape = rhs.contract().shape.dimensions;
    const auto& output_shape = output->contract().shape.dimensions;
    if (lhs_shape.size() != 2 || rhs_shape.size() != 2 || output_shape.size() != 2 ||
        lhs_shape[1] != rhs_shape[0] || output_shape[0] != lhs_shape[0] ||
        output_shape[1] != rhs_shape[1]) {
        return core::Status::error("matmul requires [M,K] * [K,N] -> [M,N]");
    }
    const core::MatmulRequest request{
        static_cast<const float*>(lhs.data()),
        static_cast<const float*>(rhs.data()),
        static_cast<float*>(output->data()),
        lhs_shape[0],
        lhs_shape[1],
        rhs_shape[1]};
    return matmul_backend_.compute(request, stream_).to_core_status();
}

core::Status CudaExecutionContext::linear(
    const CudaTensor& input,
    const CudaTensor& weights,
    CudaTensor* output) const {
    if (!valid() || output == nullptr) return core::Status::error("linear context/output invalid");
    std::size_t input_elements = 0;
    std::size_t output_elements = 0;
    if (!input.contract().shape.checked_numel(&input_elements).is_ok() ||
        !output->contract().shape.checked_numel(&output_elements).is_ok()) {
        return core::Status::error("linear input/output element count is invalid");
    }
    const auto& weight_shape = weights.contract().shape.dimensions;
    if (weight_shape != std::vector<std::size_t>{input_elements, output_elements}) {
        return core::Status::error("linear weights require [input_elements,output_elements]");
    }
    const core::MatmulRequest request{
        static_cast<const float*>(input.data()),
        static_cast<const float*>(weights.data()),
        static_cast<float*>(output->data()), 1, input_elements, output_elements};
    return matmul_backend_.compute(request, stream_).to_core_status();
}

core::Status CudaExecutionContext::linear_batch(
    const CudaTensor& input,
    const CudaLinearWeight& weights,
    CudaTensor* output) {
    if (!valid() || !weights.valid() || output == nullptr) {
        return core::Status::error("batch linear context, weights or output invalid");
    }
    const auto& input_shape = input.contract().shape.dimensions;
    const auto& output_shape = output->contract().shape.dimensions;
    if (input_shape.size() < 2 || output_shape.size() < 2 ||
        input_shape[0] != output_shape[0]) {
        return core::Status::error("batch linear requires matching leading B dimension");
    }
    std::size_t input_elements = 0;
    std::size_t output_elements = 0;
    if (!input.contract().shape.checked_numel(&input_elements).is_ok() ||
        !output->contract().shape.checked_numel(&output_elements).is_ok() ||
        input_elements % input_shape[0] != 0 ||
        output_elements % output_shape[0] != 0) {
        return core::Status::error("batch linear element count is invalid");
    }
    const std::size_t input_width = input_elements / input_shape[0];
    const std::size_t output_width = output_elements / output_shape[0];
    if (weights.kind() == CudaLinearWeight::Kind::Float32) {
        const CudaTensor* matrix = weights.float32_tensor();
        if (matrix == nullptr || matrix->contract().shape.dimensions !=
            std::vector<std::size_t>{input_width, output_width}) {
            return core::Status::error("batch Float32 Linear weight shape is invalid");
        }
        const core::MatmulRequest request{
            static_cast<const float*>(input.data()),
            static_cast<const float*>(matrix->data()),
            static_cast<float*>(output->data()),
            input_shape[0], input_width, output_width};
        return matmul_backend_.compute(request, stream_).to_core_status();
    }
    return quantized_linear_batch(*weights.quantized_matrix(), input, output);
}

core::Status CudaExecutionContext::linear(
    const CudaTensor& input,
    const CudaLinearWeight& weights,
    CudaTensor* output) {
    if (!weights.valid()) return core::Status::error("linear weight view is invalid");
    if (weights.kind() == CudaLinearWeight::Kind::Float32) {
        return linear(input, *weights.float32_tensor(), output);
    }
    return quantized_linear(*weights.quantized_matrix(), input, output);
}

core::Status CudaExecutionContext::quantized_linear(
    const CudaQuantizedMatrix& weights,
    const CudaTensor& input,
    CudaTensor* output) {
    if (!valid()) return core::Status::error("CudaExecutionContext is invalid");
    if (weights.empty()) return core::Status::error("quantized weights are empty");
    if (output == nullptr) return core::Status::error("quantized_linear output is null");
    const core::Status input_status = validate_tensor(input, "quantized_linear input");
    if (!input_status.is_ok()) return input_status;
    const core::Status output_status = validate_tensor(*output, "quantized_linear output");
    if (!output_status.is_ok()) return output_status;
    std::size_t input_elements = 0;
    std::size_t output_elements = 0;
    if (!input.contract().shape.checked_numel(&input_elements).is_ok() ||
        !output->contract().shape.checked_numel(&output_elements).is_ok() ||
        input_elements != weights.cols() || output_elements != weights.rows()) {
        return core::Status::error(
            "quantized_linear requires input/output element counts matching weights");
    }
    return cuda_quantized_gemv_backend().compute(
        weights,
        static_cast<const float*>(input.data()),
        input_elements,
        static_cast<float*>(output->data()),
        output_elements,
        stream_).to_core_status();
}

core::Status CudaExecutionContext::quantized_linear_batch(
    const CudaQuantizedMatrix& weights,
    const CudaTensor& input,
    CudaTensor* output) {
    if (!valid()) return core::Status::error("CudaExecutionContext is invalid");
    if (weights.empty()) return core::Status::error("batch quantized weights are empty");
    if (output == nullptr) return core::Status::error("batch quantized output is null");
    const core::Status input_status = validate_tensor(input, "batch quantized input");
    if (!input_status.is_ok()) return input_status;
    const core::Status output_status = validate_tensor(*output, "batch quantized output");
    if (!output_status.is_ok()) return output_status;
    const auto& input_shape = input.contract().shape.dimensions;
    const auto& output_shape = output->contract().shape.dimensions;
    if (input_shape.size() < 2 || output_shape.size() < 2 ||
        input_shape[0] != output_shape[0]) {
        return core::Status::error(
            "batch quantized linear requires matching leading B dimension");
    }
    std::size_t input_elements = 0;
    std::size_t output_elements = 0;
    if (!input.contract().shape.checked_numel(&input_elements).is_ok() ||
        !output->contract().shape.checked_numel(&output_elements).is_ok() ||
        input_elements / input_shape[0] != weights.cols() ||
        output_elements / output_shape[0] != weights.rows()) {
        return core::Status::error(
            "batch quantized linear row widths do not match weights");
    }
    return cuda_quantized_gemv_backend().compute_batch(
        weights,
        static_cast<const float*>(input.data()),
        input_shape[0],
        weights.cols(),
        static_cast<float*>(output->data()),
        weights.rows(),
        stream_,
        quantized_gemv_path_).to_core_status();
}

core::Status CudaExecutionContext::embedding_lookup(
    const CudaQuantizedMatrix& table,
    std::size_t row,
    CudaTensor* output) const {
    if (!valid() || table.empty() || output == nullptr) {
        return core::Status::error("embedding lookup context/table/output is invalid");
    }
    const core::Status output_status = validate_tensor(*output, "embedding output");
    if (!output_status.is_ok()) return output_status;
    std::size_t output_elements = 0;
    if (!output->contract().shape.checked_numel(&output_elements).is_ok() ||
        output_elements != table.cols()) {
        return core::Status::error("embedding output width does not match table");
    }
    return cuda_quantized_dequantize_backend().dequantize_row(
        table,
        row,
        static_cast<float*>(output->data()),
        output_elements,
        stream_).to_core_status();
}

core::Status CudaExecutionContext::dequantize(
    const CudaQuantizedMatrix& matrix,
    CudaTensor* output) const {
    if (!valid() || matrix.empty() || output == nullptr) {
        return core::Status::error("dequantize context, matrix or output is invalid");
    }
    const core::Status output_status = validate_tensor(*output, "dequantize output");
    if (!output_status.is_ok()) return output_status;
    std::size_t elements = 0;
    if (!output->contract().shape.checked_numel(&elements).is_ok() ||
        elements != matrix.rows() * matrix.cols()) {
        return core::Status::error("dequantize output size does not match matrix");
    }
    return cuda_quantized_dequantize_backend().dequantize(
        matrix, static_cast<float*>(output->data()), elements, stream_)
        .to_core_status();
}

core::Status CudaExecutionContext::prepare_mla(
    const CudaTensor& query,
    const CudaTensor& compressed_kv,
    CudaTensor* query_nope,
    CudaTensor* query_rope,
    CudaTensor* latent,
    CudaTensor* key_rope) const {
    if (!valid() || query_nope == nullptr || query_rope == nullptr ||
        latent == nullptr || key_rope == nullptr) {
        return core::Status::error("MLA prepare context or output is invalid");
    }
    for (const auto& item : {
             std::pair<const CudaTensor*, const char*>{&query, "MLA query"},
             {&compressed_kv, "MLA compressed K/V"},
             {query_nope, "MLA query nope"}, {query_rope, "MLA query rope"},
             {latent, "MLA latent"}, {key_rope, "MLA key rope"}}) {
        const core::Status status = validate_tensor(*item.first, item.second);
        if (!status.is_ok()) return status;
    }
    const auto& query_shape = query.contract().shape.dimensions;
    const auto& nope_shape = query_nope->contract().shape.dimensions;
    const auto& rope_shape = query_rope->contract().shape.dimensions;
    const auto& latent_shape = latent->contract().shape.dimensions;
    const auto& key_shape = key_rope->contract().shape.dimensions;
    const auto& compressed_shape = compressed_kv.contract().shape.dimensions;
    if (query_shape.size() != 2 || nope_shape.size() != 2 ||
        rope_shape.size() != 3 || rope_shape[0] != 1 ||
        latent_shape.size() != 1 || key_shape.size() != 3 ||
        key_shape[0] != 1 || key_shape[1] != 1 ||
        compressed_shape.size() != 1 || query_shape[0] != nope_shape[0] ||
        query_shape[0] != rope_shape[1] ||
        query_shape[1] != nope_shape[1] + rope_shape[2] ||
        key_shape[2] != rope_shape[2] ||
        compressed_shape[0] != latent_shape[0] + key_shape[2]) {
        return core::Status::error("MLA prepare Tensor shapes are inconsistent");
    }
    const CudaMlaPrepareRequest request{
        static_cast<const float*>(query.data()),
        static_cast<const float*>(compressed_kv.data()),
        static_cast<float*>(query_nope->data()),
        static_cast<float*>(query_rope->data()),
        static_cast<float*>(latent->data()),
        static_cast<float*>(key_rope->data()),
        query_shape[0], nope_shape[1], rope_shape[2], latent_shape[0]};
    return cuda_mla_attention_backend().prepare(request, stream_).to_core_status();
}

core::Status CudaExecutionContext::mla_attention(
    const CudaTensor& query_nope,
    const CudaTensor& query_rope,
    const CudaTensor& kv_b,
    const CudaMlaCache& cache,
    CudaTensor* probabilities,
    CudaTensor* output,
    float scale,
    CudaMlaComputePath path) const {
    if (!valid() || probabilities == nullptr || output == nullptr) {
        return core::Status::error("MLA attention context or output is invalid");
    }
    const auto& nope_shape = query_nope.contract().shape.dimensions;
    const auto& rope_shape = query_rope.contract().shape.dimensions;
    const auto& kv_shape = kv_b.contract().shape.dimensions;
    const auto& probability_shape = probabilities->contract().shape.dimensions;
    const auto& output_shape = output->contract().shape.dimensions;
    if (nope_shape.size() != 2 || rope_shape.size() != 3 ||
        rope_shape[0] != 1 || kv_shape.size() != 2 ||
        probability_shape.size() != 2 || output_shape.size() != 2 ||
        probability_shape[0] != nope_shape[0] ||
        probability_shape[1] < cache.size_tokens() ||
        nope_shape[0] != rope_shape[1] ||
        nope_shape[0] != output_shape[0] || kv_shape[1] != cache.latent_dimension() ||
        rope_shape[2] != cache.rope_dimension() ||
        kv_shape[0] != nope_shape[0] * (nope_shape[1] + output_shape[1])) {
        return core::Status::error("MLA attention Tensor shapes are inconsistent");
    }
    const CudaMlaAttentionRequest request{
        static_cast<const float*>(query_nope.data()),
        static_cast<const float*>(query_rope.data()),
        static_cast<const float*>(kv_b.data()),
        static_cast<float*>(probabilities->data()),
        static_cast<float*>(output->data()), nope_shape[0], nope_shape[1],
        rope_shape[2], output_shape[1], kv_shape[1], probability_shape[1],
        scale, 1, cache.size_tokens() - 1};
    const auto& backend = cuda_mla_attention_backend();
    return (path == CudaMlaComputePath::AbsorbedThreeStage
        ? backend.compute_absorbed(request, cache, stream_)
        : backend.compute(request, cache, stream_)).to_core_status();
}

core::Status CudaExecutionContext::prepare_mla_batch(
    const CudaTensor& query,
    const CudaTensor& compressed_kv,
    CudaTensor* query_nope,
    CudaTensor* query_rope,
    CudaTensor* latent,
    CudaTensor* key_rope) const {
    if (!valid() || query_nope == nullptr || query_rope == nullptr ||
        latent == nullptr || key_rope == nullptr) {
        return core::Status::error(
            "MLA batch prepare context or output is invalid");
    }
    const auto& query_shape = query.contract().shape.dimensions;
    const auto& compressed_shape = compressed_kv.contract().shape.dimensions;
    const auto& nope_shape = query_nope->contract().shape.dimensions;
    const auto& rope_shape = query_rope->contract().shape.dimensions;
    const auto& latent_shape = latent->contract().shape.dimensions;
    const auto& key_shape = key_rope->contract().shape.dimensions;
    if (query_shape.size() != 3 || compressed_shape.size() != 2 ||
        nope_shape.size() != 3 || rope_shape.size() != 3 ||
        latent_shape.size() != 2 || key_shape.size() != 3 ||
        query_shape[0] == 0 || query_shape[0] != compressed_shape[0] ||
        query_shape[0] != nope_shape[0] ||
        query_shape[0] != rope_shape[0] ||
        query_shape[0] != latent_shape[0] ||
        query_shape[0] != key_shape[0] || key_shape[1] != 1 ||
        query_shape[1] != nope_shape[1] ||
        query_shape[1] != rope_shape[1] ||
        query_shape[2] != nope_shape[2] + rope_shape[2] ||
        compressed_shape[1] != latent_shape[1] + key_shape[2] ||
        key_shape[2] != rope_shape[2]) {
        return core::Status::error(
            "MLA batch prepare Tensor shapes are inconsistent");
    }
    const CudaMlaPrepareRequest request{
        static_cast<const float*>(query.data()),
        static_cast<const float*>(compressed_kv.data()),
        static_cast<float*>(query_nope->data()),
        static_cast<float*>(query_rope->data()),
        static_cast<float*>(latent->data()),
        static_cast<float*>(key_rope->data()),
        query_shape[1], nope_shape[2], rope_shape[2], latent_shape[1],
        query_shape[0]};
    return cuda_mla_attention_backend().prepare(request, stream_)
        .to_core_status();
}

core::Status CudaExecutionContext::mla_attention_batch(
    const CudaTensor& query_nope,
    const CudaTensor& query_rope,
    const CudaTensor& kv_b,
    const CudaMlaCache& cache,
    std::size_t query_position_offset,
    CudaTensor* probabilities,
    CudaTensor* output,
    float scale,
    CudaMlaComputePath path) const {
    if (!valid() || probabilities == nullptr || output == nullptr) {
        return core::Status::error(
            "MLA batch attention context or output is invalid");
    }
    const auto& nope_shape = query_nope.contract().shape.dimensions;
    const auto& rope_shape = query_rope.contract().shape.dimensions;
    const auto& kv_shape = kv_b.contract().shape.dimensions;
    const auto& probability_shape = probabilities->contract().shape.dimensions;
    const auto& output_shape = output->contract().shape.dimensions;
    if (nope_shape.size() != 3 || rope_shape.size() != 3 ||
        kv_shape.size() != 2 || probability_shape.size() != 3 ||
        output_shape.size() != 3 || nope_shape[0] == 0 ||
        rope_shape[0] != nope_shape[0] || rope_shape[1] != nope_shape[1] ||
        probability_shape[0] != nope_shape[0] ||
        probability_shape[1] != nope_shape[1] ||
        probability_shape[2] < cache.size_tokens() ||
        output_shape[0] != nope_shape[0] ||
        output_shape[1] != nope_shape[1] ||
        kv_shape[1] != cache.latent_dimension() ||
        rope_shape[2] != cache.rope_dimension() ||
        kv_shape[0] != nope_shape[1] *
            (nope_shape[2] + output_shape[2])) {
        return core::Status::error(
            "MLA batch attention Tensor shapes are inconsistent");
    }
    const CudaMlaAttentionRequest request{
        static_cast<const float*>(query_nope.data()),
        static_cast<const float*>(query_rope.data()),
        static_cast<const float*>(kv_b.data()),
        static_cast<float*>(probabilities->data()),
        static_cast<float*>(output->data()),
        nope_shape[1], nope_shape[2], rope_shape[2], output_shape[2],
        kv_shape[1], probability_shape[2], scale, nope_shape[0],
        query_position_offset};
    const auto& backend = cuda_mla_attention_backend();
    return (path == CudaMlaComputePath::AbsorbedThreeStage
        ? backend.compute_absorbed(request, cache, stream_)
        : backend.compute(request, cache, stream_)).to_core_status();
}

core::Status CudaExecutionContext::paged_mla_attention(
    const CudaTensor& query_nope,
    const CudaTensor& query_rope,
    const CudaTensor& kv_b,
    const CudaPagedMlaCache& cache,
    std::size_t layer,
    const std::vector<std::size_t>& physical_blocks,
    std::size_t token_count,
    CudaTensor* probabilities,
    CudaTensor* output,
    float scale,
    CudaMlaComputePath path) {
    if (!valid() || probabilities == nullptr || output == nullptr) {
        return core::Status::error(
            "paged MLA attention context or output is invalid");
    }
    const auto& nope_shape = query_nope.contract().shape.dimensions;
    const auto& rope_shape = query_rope.contract().shape.dimensions;
    const auto& kv_shape = kv_b.contract().shape.dimensions;
    const auto& probability_shape = probabilities->contract().shape.dimensions;
    const auto& output_shape = output->contract().shape.dimensions;
    if (!cache.valid() || layer >= cache.layer_count() || token_count == 0 ||
        physical_blocks.empty() || nope_shape.size() != 2 ||
        rope_shape.size() != 3 || rope_shape[0] != 1 || kv_shape.size() != 2 ||
        probability_shape.size() != 2 || output_shape.size() != 2 ||
        probability_shape[0] != nope_shape[0] ||
        probability_shape[1] < token_count || nope_shape[0] != rope_shape[1] ||
        nope_shape[0] != output_shape[0] ||
        kv_shape[1] != cache.latent_dimension() ||
        rope_shape[2] != cache.rope_dimension() ||
        kv_shape[0] != nope_shape[0] * (nope_shape[1] + output_shape[1])) {
        return core::Status::error(
            "paged MLA attention Tensor shapes are inconsistent");
    }
    if (physical_blocks.size() >
            std::numeric_limits<std::size_t>::max() / cache.block_size() ||
        token_count > physical_blocks.size() * cache.block_size()) {
        return core::Status::error(
            "paged MLA tokens exceed the page table");
    }
    for (const std::size_t block : physical_blocks) {
        if (block >= cache.total_blocks()) {
            return core::Status::error(
                "paged MLA physical block is out of range");
        }
    }
    if (physical_blocks.size() >
        std::numeric_limits<std::size_t>::max() / sizeof(std::size_t)) {
        return core::Status::error("paged MLA page table bytes overflow");
    }
    const std::size_t page_bytes =
        physical_blocks.size() * sizeof(std::size_t);
    const core::Status workspace_status = reserve_workspace(page_bytes);
    if (!workspace_status.is_ok()) return workspace_status;
    const CudaStatus copy_status = CudaStatus::from(
        cudaMemcpyAsync(
            workspace_.data(),
            physical_blocks.data(),
            page_bytes,
            cudaMemcpyHostToDevice,
            stream_.native_handle()),
        "paged MLA page table H2D");
    if (!copy_status.is_ok()) return copy_status.to_core_status();

    const CudaMlaAttentionRequest attention{
        static_cast<const float*>(query_nope.data()),
        static_cast<const float*>(query_rope.data()),
        static_cast<const float*>(kv_b.data()),
        static_cast<float*>(probabilities->data()),
        static_cast<float*>(output->data()),
        nope_shape[0],
        nope_shape[1],
        rope_shape[2],
        output_shape[1],
        kv_shape[1],
        probability_shape[1],
        scale,
        1,
        token_count - 1};
    const CudaPagedMlaAttentionRequest request{
        attention,
        cache.latent_data(),
        cache.key_rope_data(),
        static_cast<const std::size_t*>(workspace_.data()),
        token_count,
        layer,
        cache.layer_count(),
        cache.total_blocks(),
        cache.block_size(),
        physical_blocks.size()};
    const auto& backend = cuda_mla_attention_backend();
    return (path == CudaMlaComputePath::AbsorbedThreeStage
        ? backend.compute_paged_absorbed(request, stream_)
        : backend.compute_paged(request, stream_)).to_core_status();
}

core::Status CudaExecutionContext::paged_mla_attention_batch(
    const CudaTensor& query_nope,
    const CudaTensor& query_rope,
    const CudaTensor& kv_b,
    const CudaPagedMlaCache& cache,
    std::size_t layer,
    const std::vector<std::size_t>& physical_blocks,
    std::size_t token_count,
    std::size_t query_position_offset,
    CudaTensor* probabilities,
    CudaTensor* output,
    float scale,
    CudaMlaComputePath path) {
    if (!valid() || probabilities == nullptr || output == nullptr) {
        return core::Status::error(
            "paged MLA batch attention context or output is invalid");
    }
    const auto& nope_shape = query_nope.contract().shape.dimensions;
    const auto& rope_shape = query_rope.contract().shape.dimensions;
    const auto& kv_shape = kv_b.contract().shape.dimensions;
    const auto& probability_shape = probabilities->contract().shape.dimensions;
    const auto& output_shape = output->contract().shape.dimensions;
    if (!cache.valid() || layer >= cache.layer_count() || token_count == 0 ||
        physical_blocks.empty() || nope_shape.size() != 3 ||
        rope_shape.size() != 3 || kv_shape.size() != 2 ||
        probability_shape.size() != 3 || output_shape.size() != 3 ||
        nope_shape[0] == 0 || rope_shape[0] != nope_shape[0] ||
        rope_shape[1] != nope_shape[1] ||
        probability_shape[0] != nope_shape[0] ||
        probability_shape[1] != nope_shape[1] ||
        probability_shape[2] < token_count ||
        output_shape[0] != nope_shape[0] ||
        output_shape[1] != nope_shape[1] ||
        kv_shape[1] != cache.latent_dimension() ||
        rope_shape[2] != cache.rope_dimension() ||
        kv_shape[0] != nope_shape[1] *
            (nope_shape[2] + output_shape[2]) ||
        query_position_offset >= token_count ||
        nope_shape[0] > token_count - query_position_offset) {
        return core::Status::error(
            "paged MLA batch attention Tensor shapes are inconsistent");
    }
    if (physical_blocks.size() >
            std::numeric_limits<std::size_t>::max() / cache.block_size() ||
        token_count > physical_blocks.size() * cache.block_size()) {
        return core::Status::error(
            "paged MLA batch tokens exceed the page table");
    }
    for (const std::size_t block : physical_blocks) {
        if (block >= cache.total_blocks()) {
            return core::Status::error(
                "paged MLA batch physical block is out of range");
        }
    }
    if (physical_blocks.size() >
        std::numeric_limits<std::size_t>::max() / sizeof(std::size_t)) {
        return core::Status::error(
            "paged MLA batch page table bytes overflow");
    }
    const std::size_t page_bytes =
        physical_blocks.size() * sizeof(std::size_t);
    const core::Status workspace_status = reserve_workspace(page_bytes);
    if (!workspace_status.is_ok()) return workspace_status;
    const CudaStatus copy_status = CudaStatus::from(cudaMemcpyAsync(
        workspace_.data(), physical_blocks.data(), page_bytes,
        cudaMemcpyHostToDevice, stream_.native_handle()),
        "paged MLA batch page table H2D");
    if (!copy_status.is_ok()) return copy_status.to_core_status();

    const CudaMlaAttentionRequest attention{
        static_cast<const float*>(query_nope.data()),
        static_cast<const float*>(query_rope.data()),
        static_cast<const float*>(kv_b.data()),
        static_cast<float*>(probabilities->data()),
        static_cast<float*>(output->data()),
        nope_shape[1], nope_shape[2], rope_shape[2], output_shape[2],
        kv_shape[1], probability_shape[2], scale, nope_shape[0],
        query_position_offset};
    const CudaPagedMlaAttentionRequest request{
        attention,
        cache.latent_data(),
        cache.key_rope_data(),
        static_cast<const std::size_t*>(workspace_.data()),
        token_count,
        layer,
        cache.layer_count(),
        cache.total_blocks(),
        cache.block_size(),
        physical_blocks.size()};
    const auto& backend = cuda_mla_attention_backend();
    return (path == CudaMlaComputePath::AbsorbedThreeStage
        ? backend.compute_paged_absorbed(request, stream_)
        : backend.compute_paged(request, stream_)).to_core_status();
}

core::Status CudaExecutionContext::append_mla(
    CudaMlaCache* cache,
    const CudaTensor& latent,
    const CudaTensor& key_rope) const {
    if (!valid() || cache == nullptr) {
        return core::Status::error("MLA append context or cache is invalid");
    }
    return cache->append(latent, key_rope, stream_);
}

core::Status CudaExecutionContext::append_mla_batch(
    CudaMlaCache* cache,
    const CudaTensor& latent,
    const CudaTensor& key_rope) const {
    if (!valid() || cache == nullptr) {
        return core::Status::error(
            "MLA batch append context or cache is invalid");
    }
    return cache->append_batch(latent, key_rope, stream_);
}

core::Status CudaExecutionContext::write_paged_mla(
    CudaPagedMlaCache* cache,
    std::size_t layer,
    std::size_t physical_block,
    std::size_t slot,
    const CudaTensor& latent,
    const CudaTensor& key_rope) const {
    if (!valid() || cache == nullptr) {
        return core::Status::error(
            "paged MLA write context or cache is invalid");
    }
    return cache->write_token(
        layer, physical_block, slot, latent, key_rope, stream_);
}

core::Status CudaExecutionContext::write_paged_mla_batch(
    CudaPagedMlaCache* cache,
    std::size_t layer,
    const std::vector<std::size_t>& physical_blocks,
    std::size_t start_position,
    const CudaTensor& latent,
    const CudaTensor& key_rope) const {
    if (!valid() || cache == nullptr) {
        return core::Status::error(
            "paged MLA batch write context or cache is invalid");
    }
    return cache->write_tokens(
        layer, physical_blocks, start_position, latent, key_rope, stream_);
}

core::Status CudaExecutionContext::rope(
    CudaTensor* values,
    std::size_t position_offset,
    std::size_t rotary_dimension,
    float frequency_base,
    core::RotaryPairing pairing,
    const core::YarnRotaryScaling& yarn) const {
    if (!valid()) return core::Status::error("CudaExecutionContext is invalid");
    if (values == nullptr) return core::Status::error("RoPE tensor is null");
    const core::Status tensor_status = validate_tensor(*values, "RoPE tensor");
    if (!tensor_status.is_ok()) return tensor_status;
    const auto& shape = values->contract().shape.dimensions;
    if (shape.size() != 3) {
        return core::Status::error("RoPE tensor requires [tokens,heads,head_dim]");
    }
    const core::RotaryEmbeddingRequest request{
        static_cast<float*>(values->data()),
        shape[0],
        shape[1],
        shape[2],
        rotary_dimension,
        position_offset,
        frequency_base,
        pairing,
        yarn};
    return cuda_rope_backend().apply(request, stream_).to_core_status();
}

core::Status CudaExecutionContext::rope_batch(
    CudaTensor* values,
    const std::vector<std::size_t>& positions,
    std::size_t rotary_dimension,
    float frequency_base,
    core::RotaryPairing pairing,
    const core::YarnRotaryScaling& yarn) {
    if (!valid()) return core::Status::error("CudaExecutionContext is invalid");
    if (values == nullptr || positions.empty()) {
        return core::Status::error("batch RoPE values or positions are empty");
    }
    const core::Status tensor_status = validate_tensor(*values, "batch RoPE tensor");
    if (!tensor_status.is_ok()) return tensor_status;
    const auto& shape = values->contract().shape.dimensions;
    if (shape.size() != 3 || shape[0] != positions.size()) {
        return core::Status::error(
            "batch RoPE requires values[B,H,D] and one position per row");
    }
    if (positions.size() > std::numeric_limits<std::size_t>::max() /
                               sizeof(std::size_t)) {
        return core::Status::error("batch RoPE position byte count overflows");
    }
    const std::size_t position_bytes = positions.size() * sizeof(std::size_t);
    const core::Status workspace_status = reserve_workspace(position_bytes);
    if (!workspace_status.is_ok()) return workspace_status;
    const CudaStatus upload_status = CudaStatus::from(
        cudaMemcpyAsync(
            workspace_.data(), positions.data(), position_bytes,
            cudaMemcpyHostToDevice, stream_.native_handle()),
        "batch RoPE positions upload");
    if (!upload_status.is_ok()) return upload_status.to_core_status();
    const core::RotaryEmbeddingBatchRequest request{
        static_cast<float*>(values->data()),
        static_cast<const std::size_t*>(workspace_.data()),
        shape[0], shape[1], shape[2], rotary_dimension, frequency_base, pairing,
        yarn};
    return cuda_rope_backend().apply_batch(request, stream_).to_core_status();
}

core::Status CudaExecutionContext::gqa_attention(
    const CudaTensor& query,
    const CudaKvCache& cache,
    CudaTensor* output,
    std::size_t query_position_offset,
    bool causal,
    float scale) {
    if (!valid()) return core::Status::error("CudaExecutionContext is invalid");
    if (output == nullptr) return core::Status::error("GQA output is null");
    const core::Status query_status = validate_tensor(query, "GQA query");
    if (!query_status.is_ok()) return query_status;
    const core::Status output_status = validate_tensor(*output, "GQA output");
    if (!output_status.is_ok()) return output_status;
    if (!cache.valid() || cache.size_tokens() == 0) {
        return core::Status::error("GQA KV Cache is invalid or empty");
    }
    const auto& query_shape = query.contract().shape.dimensions;
    if (query_shape.size() != 3 ||
        output->contract().shape.dimensions != query_shape ||
        query_shape[2] != cache.head_dimension() ||
        query_shape[1] % cache.kv_head_count() != 0) {
        return core::Status::error(
            "GQA requires Q/output[T,Hq,Dh] and cache[Tk,Hkv,Dh], Hq divisible by Hkv");
    }
    const core::GqaAttentionRequest request{
        static_cast<const float*>(query.data()),
        cache.key_data(),
        cache.value_data(),
        static_cast<float*>(output->data()),
        query_shape[0],
        cache.size_tokens(),
        query_shape[1],
        cache.kv_head_count(),
        query_shape[2],
        query_position_offset,
        causal,
        scale};
    const core::Status request_status = request.validate();
    if (!request_status.is_ok()) return request_status;
    if (attention_path_ == CudaAttentionPath::FusedOnline) {
        return cuda_gqa_attention_backend()
            .compute_online(request, stream_)
            .to_core_status();
    }
    if (request.score_count() >
        std::numeric_limits<std::size_t>::max() / sizeof(float)) {
        return core::Status::error("GQA workspace byte count overflows");
    }
    const core::Status workspace_status =
        reserve_workspace(request.score_count() * sizeof(float));
    if (!workspace_status.is_ok()) return workspace_status;
    return cuda_gqa_attention_backend().compute(
        request, static_cast<float*>(workspace_.data()), request.score_count(),
        stream_).to_core_status();
}

void CudaExecutionContext::set_attention_path(CudaAttentionPath path) {
    attention_path_ = path;
}

CudaAttentionPath CudaExecutionContext::attention_path() const {
    return attention_path_;
}

void CudaExecutionContext::set_quantized_gemv_path(
    CudaQuantizedGemvPath path) {
    quantized_gemv_path_ = path;
}

CudaQuantizedGemvPath CudaExecutionContext::quantized_gemv_path() const {
    return quantized_gemv_path_;
}

core::Status CudaExecutionContext::paged_gqa_attention(
    const CudaTensor& query,
    const CudaPagedKvCache& cache,
    std::size_t layer,
    const std::vector<std::size_t>& physical_blocks,
    std::size_t key_tokens,
    CudaTensor* output,
    std::size_t query_position_offset,
    bool causal,
    float scale) {
    if (!valid()) return core::Status::error("CudaExecutionContext is invalid");
    if (output == nullptr) return core::Status::error("paged GQA output is null");
    const core::Status query_status = validate_tensor(query, "paged GQA query");
    if (!query_status.is_ok()) return query_status;
    const core::Status output_status = validate_tensor(*output, "paged GQA output");
    if (!output_status.is_ok()) return output_status;
    const auto& query_shape = query.contract().shape.dimensions;
    if (!cache.valid() || layer >= cache.layer_count() || key_tokens == 0 ||
        query_shape.size() != 3 || output->contract().shape.dimensions != query_shape ||
        query_shape[2] != cache.head_dimension() ||
        query_shape[1] % cache.kv_head_count() != 0 || physical_blocks.empty()) {
        return core::Status::error("paged GQA tensor, cache, or page table is invalid");
    }
    if (physical_blocks.size() >
            std::numeric_limits<std::size_t>::max() / cache.block_size() ||
        key_tokens > physical_blocks.size() * cache.block_size()) {
        return core::Status::error("paged GQA key tokens exceed the page table");
    }
    for (const std::size_t block : physical_blocks) {
        if (block >= cache.total_blocks()) {
            return core::Status::error("paged GQA physical block is out of range");
        }
    }
    CudaPagedGqaAttentionRequest request{
        static_cast<const float*>(query.data()),
        cache.key_data(),
        cache.value_data(),
        physical_blocks.data(),
        static_cast<float*>(output->data()),
        query_shape[0],
        key_tokens,
        query_shape[1],
        cache.kv_head_count(),
        query_shape[2],
        query_position_offset,
        layer,
        cache.layer_count(),
        cache.total_blocks(),
        cache.block_size(),
        physical_blocks.size(),
        causal,
        scale};
    const core::Status request_status = request.validate();
    if (!request_status.is_ok()) return request_status;
    if (request.score_count() >
        std::numeric_limits<std::size_t>::max() / sizeof(float)) {
        return core::Status::error("paged GQA score workspace overflows");
    }
    if (physical_blocks.size() >
        std::numeric_limits<std::size_t>::max() / sizeof(std::size_t)) {
        return core::Status::error("paged GQA page table bytes overflow");
    }
    const std::size_t page_bytes = physical_blocks.size() * sizeof(std::size_t);
    const std::size_t score_bytes = request.score_count() * sizeof(float);
    if (page_bytes > std::numeric_limits<std::size_t>::max() - score_bytes) {
        return core::Status::error("paged GQA total workspace overflows");
    }
    const std::size_t required_workspace_bytes =
        attention_path_ == CudaAttentionPath::FusedOnline
        ? page_bytes
        : page_bytes + score_bytes;
    const core::Status workspace_status = reserve_workspace(required_workspace_bytes);
    if (!workspace_status.is_ok()) return workspace_status;
    auto* workspace_bytes = static_cast<unsigned char*>(workspace_.data());
    const CudaStatus copy_status = CudaStatus::from(
        cudaMemcpyAsync(
            workspace_bytes,
            physical_blocks.data(),
            page_bytes,
            cudaMemcpyHostToDevice,
            stream_.native_handle()),
        "paged GQA page table H2D");
    if (!copy_status.is_ok()) return copy_status.to_core_status();
    request.physical_blocks = reinterpret_cast<const std::size_t*>(workspace_bytes);
    if (attention_path_ == CudaAttentionPath::FusedOnline) {
        return cuda_gqa_attention_backend()
            .compute_paged_online(request, stream_)
            .to_core_status();
    }
    auto* scores = reinterpret_cast<float*>(workspace_bytes + page_bytes);
    return cuda_gqa_attention_backend().compute_paged(
        request, scores, request.score_count(), stream_).to_core_status();
}

core::Status CudaExecutionContext::append_kv(
    CudaKvCache* cache,
    const CudaTensor& key,
    const CudaTensor& value) const {
    if (!valid()) return core::Status::error("CudaExecutionContext is invalid");
    if (cache == nullptr) return core::Status::error("KV Cache is null");
    return cache->append(key, value, stream_);
}

core::Status CudaExecutionContext::write_paged_kv(
    CudaPagedKvCache* cache,
    std::size_t layer,
    std::size_t physical_block,
    std::size_t slot,
    const CudaTensor& key,
    const CudaTensor& value) const {
    if (!valid()) return core::Status::error("CudaExecutionContext is invalid");
    if (cache == nullptr) return core::Status::error("paged KV cache is null");
    return cache->write_token(
        layer, physical_block, slot, key, value, stream_);
}

core::Status CudaExecutionContext::download_kv(
    const CudaKvCache& cache,
    float* key,
    float* value,
    std::size_t token_count) const {
    if (!valid()) return core::Status::error("CudaExecutionContext is invalid");
    return cache.copy_to_host_async(key, value, token_count, stream_);
}

core::Status CudaExecutionContext::reserve_workspace(std::size_t bytes) {
    if (!valid()) return core::Status::error("CudaExecutionContext is invalid");
    return workspace_.reserve(bytes).to_core_status();
}

core::Status CudaExecutionContext::synchronize() const {
    if (!valid()) return core::Status::error("CudaExecutionContext is invalid");
    return stream_.synchronize().to_core_status();
}

std::size_t CudaExecutionContext::workspace_capacity_bytes() const {
    return workspace_.capacity_bytes();
}

std::size_t CudaExecutionContext::workspace_allocation_count() const {
    return workspace_.allocation_count();
}

bool CudaExecutionContext::valid() const {
    return stream_.valid() && matmul_backend_.valid();
}

}  // namespace cpp_llm_engine::cuda
