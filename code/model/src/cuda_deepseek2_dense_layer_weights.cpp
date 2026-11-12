// DeepSeek2 CUDA Model：实现真实稠密层权重的格式校验与原子上传。
#include "cpp_llm_engine/model/cuda_deepseek2_dense_layer_weights.h"

#include "cpp_llm_engine/cuda/cuda_execution_context.h"
#include "real_layer_weights.h"

#include <cstddef>
#include <utility>

namespace cpp_llm_engine::model {
namespace {

core::Status upload_norm(cuda::CudaExecutionContext* context,
                         const chapter01::Tensor& host,
                         cuda::CudaTensor* output) {
    if (host.dtype() != chapter01::DType::Float32 || host.shape().size() != 1 ||
        host.shape()[0] <= 0 || host.raw_data() == nullptr) {
        return core::Status::error(
            "CUDA DeepSeek2 norm must be a non-empty Float32 vector");
    }
    const auto width = static_cast<std::size_t>(host.shape()[0]);
    core::Status status = cuda::CudaTensor::create_contiguous(
        {width}, core::DType::Float32, output);
    if (!status.is_ok()) return status;
    return context->upload(output, host.raw_data(), host.bytes());
}

core::WeightFormat weight_format(chapter01::DType dtype) {
    if (dtype == chapter01::DType::Q2_K) return core::WeightFormat::Q2K;
    if (dtype == chapter01::DType::Q3_K) return core::WeightFormat::Q3K;
    if (dtype == chapter01::DType::IQ4_NL) return core::WeightFormat::IQ4_NL;
    return core::WeightFormat::Unknown;
}

core::Status upload_linear(cuda::CudaExecutionContext* context,
                           const chapter01::Tensor& host,
                           cuda::CudaQuantizedMatrix* output) {
    if (host.shape().size() != 2 || host.shape()[0] <= 0 ||
        host.shape()[1] <= 0 || host.raw_data() == nullptr) {
        return core::Status::error(
            "CUDA DeepSeek2 Linear requires a non-empty 2D tensor");
    }
    const core::WeightFormat format = weight_format(host.dtype());
    if (format == core::WeightFormat::Unknown) {
        return core::Status::error(
            "CUDA DeepSeek2 Linear requires Q2_K, Q3_K, or IQ4_NL");
    }
    const auto cols = static_cast<std::size_t>(host.shape()[0]);
    const auto rows = static_cast<std::size_t>(host.shape()[1]);
    core::Status status = cuda::CudaQuantizedMatrix::create(
        rows, cols, format, output);
    if (!status.is_ok()) return status;
    return context->upload(output, host.raw_data(), host.bytes());
}

}  // namespace

bool CudaDeepSeek2DenseLayerWeightView::valid() const {
    return attention_norm != nullptr && !attention_norm->empty() &&
        query.valid() && kv_a_mqa.valid() && kv_a_norm != nullptr &&
        !kv_a_norm->empty() && kv_b.valid() && kv_b_dequantized != nullptr &&
        !kv_b_dequantized->empty() && attention_output.valid() &&
        ffn_norm != nullptr && !ffn_norm->empty() && ffn_gate.valid() &&
        ffn_up.valid() && ffn_down.valid();
}

core::Status CudaDeepSeek2DenseLayerWeights::upload(
    cuda::CudaExecutionContext* context,
    const chapter01::RealDeepSeek2DenseLayerWeights& host,
    CudaDeepSeek2DenseLayerWeights* output) {
    if (context == nullptr || output == nullptr || !host.valid() ||
        !host.uses_combined_kv_b()) {
        return core::Status::error("CUDA DeepSeek2 layer upload input is invalid");
    }
    CudaDeepSeek2DenseLayerWeights replacement;
#define UPLOAD(destination, source, function) do { \
    const auto status = function(context, source, &destination); \
    if (!status.is_ok()) return status; } while (false)
    UPLOAD(replacement.attention_norm_, host.attn_norm, upload_norm);
    UPLOAD(replacement.query_, host.attn_q, upload_linear);
    UPLOAD(replacement.kv_a_mqa_, host.attn_kv_a_mqa, upload_linear);
    UPLOAD(replacement.kv_a_norm_, host.attn_kv_a_norm, upload_norm);
    UPLOAD(replacement.kv_b_, host.attn_kv_b, upload_linear);
    core::Status kv_b_status = cuda::CudaTensor::create_contiguous(
        {replacement.kv_b_.rows(), replacement.kv_b_.cols()},
        core::DType::Float32, &replacement.kv_b_dequantized_);
    if (!kv_b_status.is_ok()) return kv_b_status;
    kv_b_status = context->dequantize(
        replacement.kv_b_, &replacement.kv_b_dequantized_);
    if (!kv_b_status.is_ok()) return kv_b_status;
    UPLOAD(replacement.attention_output_, host.attn_output, upload_linear);
    UPLOAD(replacement.ffn_norm_, host.ffn_norm, upload_norm);
    UPLOAD(replacement.ffn_gate_, host.ffn_gate, upload_linear);
    UPLOAD(replacement.ffn_up_, host.ffn_up, upload_linear);
    UPLOAD(replacement.ffn_down_, host.ffn_down, upload_linear);
#undef UPLOAD
    const core::Status sync_status = context->synchronize();
    if (!sync_status.is_ok()) return sync_status;
    *output = std::move(replacement);
    return core::Status::ok();
}

CudaDeepSeek2DenseLayerWeightView
CudaDeepSeek2DenseLayerWeights::view() const {
    return {&attention_norm_, cuda::CudaLinearWeight::quantized(&query_),
        cuda::CudaLinearWeight::quantized(&kv_a_mqa_), &kv_a_norm_,
        cuda::CudaLinearWeight::quantized(&kv_b_),
        &kv_b_dequantized_,
        cuda::CudaLinearWeight::quantized(&attention_output_), &ffn_norm_,
        cuda::CudaLinearWeight::quantized(&ffn_gate_),
        cuda::CudaLinearWeight::quantized(&ffn_up_),
        cuda::CudaLinearWeight::quantized(&ffn_down_)};
}

bool CudaDeepSeek2DenseLayerWeights::valid() const {
    return view().valid();
}

}  // namespace cpp_llm_engine::model
