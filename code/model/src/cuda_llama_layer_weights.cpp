// Llama CUDA Model：实现真实 layer 权重的格式校验与显存上传。
// 职责：把 GGUF `[input,output]` shape 转换为 CUDA rows=output、cols=input。
// 契约：仅接受 F32 norm 与 Q4_K/Q6_K Linear，不做永久 Float32 反量化。
// 验证：cuda_llama_layer_weights_test 在真实 GPU 上执行。
#include "cpp_llm_engine/model/cuda_llama_layer_weights.h"

#include "cpp_llm_engine/cuda/cuda_execution_context.h"
#include "real_layer_weights.h"

#include <cstddef>
#include <utility>
#include <vector>

namespace cpp_llm_engine::model {
namespace {

core::Status upload_norm(
    cuda::CudaExecutionContext* context,
    const chapter01::Tensor& host,
    cuda::CudaTensor* output) {
    if (host.dtype() != chapter01::DType::Float32 || host.shape().size() != 1 ||
        host.shape()[0] <= 0 || host.raw_data() == nullptr) {
        return core::Status::error("CUDA Llama norm must be a non-empty Float32 vector");
    }
    const auto width = static_cast<std::size_t>(host.shape()[0]);
    core::Status status = cuda::CudaTensor::create_contiguous(
        {width}, core::DType::Float32, output);
    if (!status.is_ok()) return status;
    return context->upload(output, host.raw_data(), host.bytes());
}

core::Status upload_linear(
    cuda::CudaExecutionContext* context,
    const chapter01::Tensor& host,
    cuda::CudaQuantizedMatrix* output) {
    if (host.shape().size() != 2 || host.shape()[0] <= 0 || host.shape()[1] <= 0 ||
        host.raw_data() == nullptr) {
        return core::Status::error("CUDA Llama Linear requires a non-empty 2D tensor");
    }
    core::WeightFormat format = core::WeightFormat::Unknown;
    if (host.dtype() == chapter01::DType::Q4_K) format = core::WeightFormat::Q4K;
    if (host.dtype() == chapter01::DType::Q6_K) format = core::WeightFormat::Q6K;
    if (format == core::WeightFormat::Unknown) {
        return core::Status::error("CUDA Llama Linear requires Q4_K or Q6_K");
    }
    const auto cols = static_cast<std::size_t>(host.shape()[0]);
    const auto rows = static_cast<std::size_t>(host.shape()[1]);
    core::Status status = cuda::CudaQuantizedMatrix::create(rows, cols, format, output);
    if (!status.is_ok()) return status;
    return context->upload(output, host.raw_data(), host.bytes());
}

}  // namespace

core::Status CudaLlamaLayerWeights::upload(
    cuda::CudaExecutionContext* context,
    const chapter01::RealDenseLayerWeights& host,
    CudaLlamaLayerWeights* output) {
    if (context == nullptr || output == nullptr || !host.valid()) {
        return core::Status::error("CUDA Llama layer upload input is invalid");
    }
    CudaLlamaLayerWeights replacement;
#define UPLOAD(destination, source, function) do { \
    const auto status = function(context, source, &destination); \
    if (!status.is_ok()) return status; } while (false)
    UPLOAD(replacement.attention_norm_, host.attn_norm, upload_norm);
    if (host.attn_q_norm.bytes() > 0) {
        UPLOAD(replacement.query_norm_, host.attn_q_norm, upload_norm);
        UPLOAD(replacement.key_norm_, host.attn_k_norm, upload_norm);
    }
    UPLOAD(replacement.query_, host.attn_q, upload_linear);
    UPLOAD(replacement.key_, host.attn_k, upload_linear);
    UPLOAD(replacement.value_, host.attn_v, upload_linear);
    UPLOAD(replacement.attention_output_, host.attn_output, upload_linear);
    UPLOAD(replacement.ffn_norm_, host.ffn_norm, upload_norm);
    UPLOAD(replacement.gate_, host.ffn_gate, upload_linear);
    UPLOAD(replacement.up_, host.ffn_up, upload_linear);
    UPLOAD(replacement.down_, host.ffn_down, upload_linear);
#undef UPLOAD
    const core::Status sync_status = context->synchronize();
    if (!sync_status.is_ok()) return sync_status;
    *output = std::move(replacement);
    return core::Status::ok();
}

cuda::CudaDecoderLayerWeights CudaLlamaLayerWeights::decoder_weights() const {
    return {&attention_norm_, cuda::CudaLinearWeight::quantized(&query_),
        cuda::CudaLinearWeight::quantized(&key_), cuda::CudaLinearWeight::quantized(&value_),
        cuda::CudaLinearWeight::quantized(&attention_output_), &ffn_norm_,
        cuda::CudaLinearWeight::quantized(&gate_), cuda::CudaLinearWeight::quantized(&up_),
        cuda::CudaLinearWeight::quantized(&down_),
        query_norm_.empty() ? nullptr : &query_norm_,
        key_norm_.empty() ? nullptr : &key_norm_};
}

bool CudaLlamaLayerWeights::valid() const {
    const auto weights = decoder_weights();
    const bool qk_norms_match = query_norm_.empty() == key_norm_.empty();
    return qk_norms_match && !attention_norm_.empty() && !ffn_norm_.empty() &&
        weights.query.valid() &&
        weights.key.valid() && weights.value.valid() && weights.attention_output.valid() &&
        weights.gate.valid() && weights.up.valid() && weights.down.valid();
}

const cuda::CudaQuantizedMatrix& CudaLlamaLayerWeights::query() const { return query_; }
const cuda::CudaQuantizedMatrix& CudaLlamaLayerWeights::down() const { return down_; }

}  // namespace cpp_llm_engine::model
