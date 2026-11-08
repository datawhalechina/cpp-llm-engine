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
#include "cpp_llm_engine/cuda/gqa_attention_backend.h"
#include "cpp_llm_engine/cuda/quantized_dequantize_backend.h"
#include "cpp_llm_engine/cuda/rope_backend.h"

#include <limits>
#include <string>
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
    if (weights.rows() > std::numeric_limits<std::size_t>::max() / weights.cols()) {
        return core::Status::error("quantized_linear element count overflows");
    }
    const std::size_t elements = weights.rows() * weights.cols();
    if (elements > std::numeric_limits<std::size_t>::max() / sizeof(float)) {
        return core::Status::error("quantized_linear workspace byte count overflows");
    }
    const core::Status workspace_status = reserve_workspace(elements * sizeof(float));
    if (!workspace_status.is_ok()) return workspace_status;
    const CudaStatus dequantize_status = cuda_quantized_dequantize_backend().dequantize(
        weights, static_cast<float*>(workspace_.data()), elements, stream_);
    if (!dequantize_status.is_ok()) return dequantize_status.to_core_status();
    const core::MatmulRequest request{
        static_cast<const float*>(workspace_.data()),
        static_cast<const float*>(input.data()),
        static_cast<float*>(output->data()),
        weights.rows(),
        weights.cols(),
        1};
    return matmul_backend_.compute(request, stream_).to_core_status();
}

core::Status CudaExecutionContext::rope(
    CudaTensor* values,
    std::size_t position_offset,
    std::size_t rotary_dimension,
    float frequency_base) const {
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
        frequency_base};
    return cuda_rope_backend().apply(request, stream_).to_core_status();
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

core::Status CudaExecutionContext::append_kv(
    CudaKvCache* cache,
    const CudaTensor& key,
    const CudaTensor& value) const {
    if (!valid()) return core::Status::error("CudaExecutionContext is invalid");
    if (cache == nullptr) return core::Status::error("KV Cache is null");
    return cache->append(key, value, stream_);
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
