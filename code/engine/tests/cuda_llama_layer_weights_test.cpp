// Engine CUDA：验证 Chapter01 layer 权重到 CUDA Model 权重的上传边界。
// 职责：检查 F32 norm、Q4_K/Q6_K 格式、矩阵方向、payload bytes 和失败路径。
// 通过条件：九个权重上传后视图有效，host 销毁不影响 device 所有权。
#include "cpp_llm_engine/model/cuda_llama_layer_weights.h"

#include "cpp_llm_engine/cuda/cuda_execution_context.h"
#include "real_layer_weights.h"

#include <cuda_runtime_api.h>

#include <cstddef>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#define CHECK(c) do { if (!(c)) { std::cerr << "check failed line " << __LINE__ << '\n'; return 1; } } while (false)

namespace {

chapter01::Tensor packed(
    const std::string& name,
    chapter01::DType dtype,
    std::size_t input_width,
    std::size_t output_width) {
    const std::size_t block_bytes = dtype == chapter01::DType::Q4_K ? 144 : 210;
    const std::size_t blocks_per_row = input_width / 256;
    std::vector<std::byte> bytes(
        output_width * blocks_per_row * block_bytes, std::byte{0});
    return chapter01::Tensor::from_raw_bytes(
        name,
        {static_cast<std::int64_t>(input_width),
         static_cast<std::int64_t>(output_width)},
        dtype,
        std::move(bytes));
}

chapter01::RealLlamaLayerWeights make_layer() {
    chapter01::RealLlamaLayerWeights layer;
    layer.attn_norm = chapter01::Tensor::from_vector<float>(
        "attn_norm", {256}, std::vector<float>(256, 1.0F));
    layer.ffn_norm = chapter01::Tensor::from_vector<float>(
        "ffn_norm", {256}, std::vector<float>(256, 2.0F));
    layer.attn_q = packed("q", chapter01::DType::Q4_K, 256, 256);
    layer.attn_k = packed("k", chapter01::DType::Q4_K, 256, 64);
    layer.attn_v = packed("v", chapter01::DType::Q6_K, 256, 64);
    layer.attn_output = packed("o", chapter01::DType::Q6_K, 256, 256);
    layer.ffn_gate = packed("gate", chapter01::DType::Q4_K, 256, 512);
    layer.ffn_up = packed("up", chapter01::DType::Q6_K, 256, 512);
    layer.ffn_down = packed("down", chapter01::DType::Q6_K, 512, 256);
    return layer;
}

bool matrix_matches(
    const cpp_llm_engine::cuda::CudaLinearWeight& weight,
    std::size_t rows,
    std::size_t cols,
    cpp_llm_engine::core::WeightFormat format,
    std::size_t bytes) {
    if (!weight.valid() ||
        weight.kind() != cpp_llm_engine::cuda::CudaLinearWeight::Kind::Quantized) {
        return false;
    }
    const auto* matrix = weight.quantized_matrix();
    return matrix != nullptr && matrix->rows() == rows && matrix->cols() == cols &&
        matrix->format() == format && matrix->size_bytes() == bytes;
}

}  // namespace

int main() {
    int devices = 0;
    CHECK(cudaGetDeviceCount(&devices) == cudaSuccess && devices > 0);
    cpp_llm_engine::cuda::CudaExecutionContext context;
    CHECK(cpp_llm_engine::cuda::CudaExecutionContext::create(&context).is_ok());
    chapter01::RealLlamaLayerWeights host = make_layer();
    cpp_llm_engine::model::CudaLlamaLayerWeights device;
    CHECK(cpp_llm_engine::model::CudaLlamaLayerWeights::upload(
        &context, host, &device).is_ok());
    CHECK(device.valid());
    const auto weights = device.decoder_weights();
    CHECK(weights.attention_norm != nullptr);
    CHECK(weights.attention_norm->contract().shape.dimensions ==
          std::vector<std::size_t>{256});
    CHECK(weights.ffn_norm != nullptr);
    CHECK(weights.ffn_norm->contract().shape.dimensions ==
          std::vector<std::size_t>{256});
    CHECK(matrix_matches(weights.query, 256, 256,
        cpp_llm_engine::core::WeightFormat::Q4K, 256 * 144));
    CHECK(matrix_matches(weights.key, 64, 256,
        cpp_llm_engine::core::WeightFormat::Q4K, 64 * 144));
    CHECK(matrix_matches(weights.value, 64, 256,
        cpp_llm_engine::core::WeightFormat::Q6K, 64 * 210));
    CHECK(matrix_matches(weights.attention_output, 256, 256,
        cpp_llm_engine::core::WeightFormat::Q6K, 256 * 210));
    CHECK(matrix_matches(weights.gate, 512, 256,
        cpp_llm_engine::core::WeightFormat::Q4K, 512 * 144));
    CHECK(matrix_matches(weights.up, 512, 256,
        cpp_llm_engine::core::WeightFormat::Q6K, 512 * 210));
    CHECK(matrix_matches(weights.down, 256, 512,
        cpp_llm_engine::core::WeightFormat::Q6K, 256 * 2 * 210));

    std::vector<float> attention_norm(256, 0.0F);
    std::vector<float> ffn_norm(256, 0.0F);
    CHECK(context.download(
        *weights.attention_norm, attention_norm.data(), 256 * sizeof(float)).is_ok());
    CHECK(context.download(
        *weights.ffn_norm, ffn_norm.data(), 256 * sizeof(float)).is_ok());
    CHECK(context.synchronize().is_ok());
    CHECK(attention_norm.front() == 1.0F && attention_norm.back() == 1.0F);
    CHECK(ffn_norm.front() == 2.0F && ffn_norm.back() == 2.0F);

    host = {};
    CHECK(device.valid());
    CHECK(device.decoder_weights().key.valid());

    cpp_llm_engine::model::CudaLlamaLayerWeights moved = std::move(device);
    CHECK(moved.valid());
    CHECK(!device.valid());

    chapter01::RealLlamaLayerWeights invalid = make_layer();
    invalid.attn_norm.set_dtype(chapter01::DType::Float16);
    cpp_llm_engine::model::CudaLlamaLayerWeights rejected;
    CHECK(!cpp_llm_engine::model::CudaLlamaLayerWeights::upload(
        &context, invalid, &rejected).is_ok());
    CHECK(!rejected.valid());
    CHECK(!cpp_llm_engine::model::CudaLlamaLayerWeights::upload(
        nullptr, invalid, &rejected).is_ok());
    CHECK(!cpp_llm_engine::model::CudaLlamaLayerWeights::upload(
        &context, make_layer(), nullptr).is_ok());

    invalid = make_layer();
    invalid.ffn_up.set_dtype(chapter01::DType::Float32);
    CHECK(!cpp_llm_engine::model::CudaLlamaLayerWeights::upload(
        &context, invalid, &moved).is_ok());
    CHECK(moved.valid());
    CHECK(moved.down().cols() == 512);
    return 0;
}

#undef CHECK
