// Engine CUDA：验证 DeepSeek2 稠密层 host 权重到 device 权重的上传边界。
// 职责：检查 Q2_K/Q3_K/IQ4_NL、矩阵方向、所有权和失败原子性。
#include "cpp_llm_engine/model/cuda_deepseek2_dense_layer_weights.h"
#include "cpp_llm_engine/model/cuda_deepseek2_dense_layer.h"

#include "cpp_llm_engine/cuda/cuda_execution_context.h"
#include "cpp_llm_engine/cuda/cuda_tensor.h"
#include "real_layer_weights.h"

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cmath>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#define CHECK(c) do { if (!(c)) { \
    std::cerr << "check failed line " << __LINE__ << '\n'; return 1; \
} } while (false)

namespace {

std::size_t block_bytes(chapter01::DType dtype) {
    if (dtype == chapter01::DType::Q2_K) return 84;
    if (dtype == chapter01::DType::Q3_K) return 110;
    return 18;
}

std::size_t block_elements(chapter01::DType dtype) {
    return dtype == chapter01::DType::IQ4_NL ? 32 : 256;
}

chapter01::Tensor packed(const std::string& name,
                         chapter01::DType dtype,
                         std::size_t input_width,
                         std::size_t output_width) {
    std::vector<std::byte> bytes(
        output_width * (input_width / block_elements(dtype)) * block_bytes(dtype),
        std::byte{0});
    return chapter01::Tensor::from_raw_bytes(
        name,
        {static_cast<std::int64_t>(input_width),
         static_cast<std::int64_t>(output_width)},
        dtype,
        std::move(bytes));
}

chapter01::RealDeepSeek2DenseLayerWeights make_layer() {
    chapter01::RealDeepSeek2DenseLayerWeights layer;
    layer.attn_norm = chapter01::Tensor::from_vector<float>(
        "attn_norm", {256}, std::vector<float>(256, 1.0F));
    layer.attn_q = packed("q", chapter01::DType::Q2_K, 256, 256);
    layer.attn_kv_a_mqa = packed("kv_a", chapter01::DType::Q2_K, 256, 320);
    layer.attn_kv_a_norm = chapter01::Tensor::from_vector<float>(
        "kv_a_norm", {256}, std::vector<float>(256, 0.5F));
    layer.attn_kv_b = packed("kv_b", chapter01::DType::Q2_K, 256, 384);
    layer.attn_output = packed("o", chapter01::DType::Q3_K, 256, 256);
    layer.ffn_norm = chapter01::Tensor::from_vector<float>(
        "ffn_norm", {256}, std::vector<float>(256, 2.0F));
    layer.ffn_gate = packed("gate", chapter01::DType::Q2_K, 256, 512);
    layer.ffn_up = packed("up", chapter01::DType::Q2_K, 256, 512);
    layer.ffn_down = packed("down", chapter01::DType::IQ4_NL, 512, 256);
    return layer;
}

bool matrix_matches(const cpp_llm_engine::cuda::CudaLinearWeight& weight,
                    std::size_t rows,
                    std::size_t cols,
                    cpp_llm_engine::core::WeightFormat format,
                    std::size_t bytes) {
    const auto* matrix = weight.quantized_matrix();
    return weight.valid() && matrix != nullptr && matrix->rows() == rows &&
        matrix->cols() == cols && matrix->format() == format &&
        matrix->size_bytes() == bytes;
}

}  // namespace

int main() {
    int devices = 0;
    CHECK(cudaGetDeviceCount(&devices) == cudaSuccess && devices > 0);
    cpp_llm_engine::cuda::CudaExecutionContext context;
    CHECK(cpp_llm_engine::cuda::CudaExecutionContext::create(&context).is_ok());

    chapter01::RealDeepSeek2DenseLayerWeights host = make_layer();
    cpp_llm_engine::model::CudaDeepSeek2DenseLayerWeights device;
    CHECK(cpp_llm_engine::model::CudaDeepSeek2DenseLayerWeights::upload(
        &context, host, &device).is_ok());
    CHECK(device.valid());
    const auto weights = device.view();
    CHECK(matrix_matches(weights.query, 256, 256,
        cpp_llm_engine::core::WeightFormat::Q2K, 256 * 84));
    CHECK(matrix_matches(weights.kv_a_mqa, 320, 256,
        cpp_llm_engine::core::WeightFormat::Q2K, 320 * 84));
    CHECK(matrix_matches(weights.kv_b, 384, 256,
        cpp_llm_engine::core::WeightFormat::Q2K, 384 * 84));
    CHECK(matrix_matches(weights.attention_output, 256, 256,
        cpp_llm_engine::core::WeightFormat::Q3K, 256 * 110));
    CHECK(matrix_matches(weights.ffn_down, 256, 512,
        cpp_llm_engine::core::WeightFormat::IQ4_NL, 256 * 16 * 18));

    std::vector<float> norm(256, 0.0F);
    CHECK(context.download(
        *weights.kv_a_norm, norm.data(), norm.size() * sizeof(float)).is_ok());
    CHECK(context.synchronize().is_ok());
    CHECK(norm.front() == 0.5F && norm.back() == 0.5F);

    cpp_llm_engine::model::CudaDeepSeek2DenseLayer layer;
    const cpp_llm_engine::model::CudaDeepSeek2DenseLayerConfig config{
        256, 2, 64, 64, 128, 256, 512, 1.0e-6F, 10000.0F, {},
        1.0F / std::sqrt(128.0F)};
    CHECK(cpp_llm_engine::model::CudaDeepSeek2DenseLayer::create(
        config, 2, &layer).is_ok());
    cpp_llm_engine::cuda::CudaTensor hidden;
    CHECK(cpp_llm_engine::cuda::CudaTensor::create_contiguous(
        {256}, cpp_llm_engine::core::DType::Float32, &hidden).is_ok());
    std::vector<float> hidden_host(256);
    for (std::size_t index = 0; index < hidden_host.size(); ++index) {
        hidden_host[index] = static_cast<float>(index % 17) * 0.01F;
    }
    CHECK(context.upload(
        &hidden, hidden_host.data(), hidden_host.size() * sizeof(float)).is_ok());
    CHECK(layer.forward(&context, weights, &hidden, 0).is_ok());
    CHECK(layer.forward(&context, weights, &hidden, 1).is_ok());
    std::vector<float> forwarded(256);
    CHECK(context.download(
        hidden, forwarded.data(), forwarded.size() * sizeof(float)).is_ok());
    CHECK(context.synchronize().is_ok());
    CHECK(layer.position() == 2);
    for (std::size_t index = 0; index < forwarded.size(); ++index) {
        CHECK(std::fabs(forwarded[index] - hidden_host[index]) <= 1.0e-6F);
    }
    CHECK(!layer.forward(&context, weights, &hidden, 1).is_ok());
    CHECK(layer.position() == 2);
    layer.reset();
    CHECK(layer.position() == 0);

    host = {};
    CHECK(device.valid());
    cpp_llm_engine::model::CudaDeepSeek2DenseLayerWeights moved =
        std::move(device);
    CHECK(moved.valid());
    CHECK(!device.valid());

    auto invalid = make_layer();
    invalid.attn_output.set_dtype(chapter01::DType::Q4_K);
    CHECK(!cpp_llm_engine::model::CudaDeepSeek2DenseLayerWeights::upload(
        &context, invalid, &moved).is_ok());
    CHECK(moved.valid());
    CHECK(moved.view().attention_output.quantized_matrix()->format() ==
        cpp_llm_engine::core::WeightFormat::Q3K);
    CHECK(!cpp_llm_engine::model::CudaDeepSeek2DenseLayerWeights::upload(
        nullptr, make_layer(), &moved).is_ok());
    CHECK(!cpp_llm_engine::model::CudaDeepSeek2DenseLayerWeights::upload(
        &context, make_layer(), nullptr).is_ok());
    return 0;
}

#undef CHECK
