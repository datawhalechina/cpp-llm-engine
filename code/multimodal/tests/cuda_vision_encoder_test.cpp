// 多模态 CUDA：验证多层 SigLIP Encoder 与持久 workspace。
#include "multimodal/cuda_vision_encoder.h"

#include "cpp_llm_engine/cuda/cuda_execution_context.h"

#include <cmath>
#include <cstddef>
#include <iostream>
#include <string>
#include <vector>

#define CHECK(condition)                                                        \
    do {                                                                        \
        if (!(condition)) {                                                     \
            std::cerr << "check failed: " #condition << " at line " << __LINE__ \
                      << '\n';                                                  \
            return 1;                                                           \
        }                                                                       \
    } while (false)

namespace {

multimodal::LinearWeights make_linear(
    std::size_t input_size,
    std::size_t output_size,
    float scale) {
    multimodal::LinearWeights result;
    result.input_size = input_size;
    result.output_size = output_size;
    result.weight.resize(input_size * output_size);
    result.bias.resize(output_size);
    for (std::size_t output = 0; output < output_size; ++output) {
        result.bias[output] = scale * static_cast<float>(output + 1) * 0.05F;
        for (std::size_t input = 0; input < input_size; ++input) {
            const int pattern = static_cast<int>((output * 5 + input * 3) % 9) - 4;
            result.weight[output * input_size + input] =
                scale * static_cast<float>(pattern);
        }
    }
    return result;
}

multimodal::SiglipBlockWeights make_block(float scale) {
    multimodal::SiglipBlockWeights result;
    result.layer_norm_1 = {{1.0F, 0.9F, 1.1F, 1.2F}, {0.01F, 0.0F, -0.01F, 0.02F}};
    result.query = make_linear(4, 4, scale);
    result.key = make_linear(4, 4, scale * 0.8F);
    result.value = make_linear(4, 4, scale * 0.6F);
    result.attention_output = make_linear(4, 4, scale * 0.5F);
    result.layer_norm_2 = {{0.8F, 1.2F, 1.0F, 0.9F}, {-0.01F, 0.02F, 0.0F, 0.01F}};
    result.feed_forward_input = make_linear(4, 6, scale * 0.4F);
    result.feed_forward_output = make_linear(6, 4, scale * 0.3F);
    return result;
}

}  // namespace

int main() {
    constexpr std::size_t tokens = 3;
    constexpr std::size_t hidden = 4;
    constexpr std::size_t heads = 2;
    constexpr float epsilon = 1.0e-5F;
    const std::vector<multimodal::SiglipBlockWeights> blocks{
        make_block(0.04F), make_block(-0.03F)};
    const multimodal::LayerNormWeights post_norm{
        {1.1F, 0.9F, 1.0F, 1.2F}, {0.01F, -0.02F, 0.03F, 0.0F}};
    const multimodal::FloatMatrix input{
        tokens, hidden,
        {0.2F, -0.4F, 0.6F, 0.8F,
         -0.3F, 0.7F, -0.5F, 0.1F,
         0.9F, -0.2F, 0.4F, -0.8F}};
    const multimodal::SiglipBlockWeightProvider provider =
        [&blocks](std::size_t layer, multimodal::SiglipBlockWeights* output,
                  std::string*) {
            if (layer >= blocks.size() || output == nullptr) return false;
            *output = blocks[layer];
            return true;
        };
    multimodal::FloatMatrix expected;
    std::vector<multimodal::SiglipLayerSummary> summaries;
    multimodal::SiglipEncoderStats stats;
    std::string error;
    CHECK(multimodal::run_siglip_encoder(
        input, blocks.size(), heads, epsilon, provider, post_norm,
        &expected, &summaries, &stats, &error));

    using cpp_llm_engine::core::DType;
    using cpp_llm_engine::cuda::CudaExecutionContext;
    using cpp_llm_engine::cuda::CudaTensor;
    CudaExecutionContext context;
    multimodal::CudaSiglipEncoder encoder;
    CudaTensor device_input;
    CudaTensor device_output;
    CHECK(CudaExecutionContext::create(&context).is_ok());
    CHECK(multimodal::CudaSiglipEncoder::upload(
        &context, blocks, post_norm, heads, epsilon, &encoder).is_ok());
    CHECK(encoder.block_count() == blocks.size());
    CHECK(CudaTensor::create_contiguous(
        {tokens, hidden}, DType::Float32, &device_input).is_ok());
    CHECK(CudaTensor::create_contiguous(
        {tokens, hidden}, DType::Float32, &device_output).is_ok());
    CHECK(context.upload(
        &device_input, input.values.data(),
        input.values.size() * sizeof(float)).is_ok());
    CHECK(encoder.forward(&context, device_input, &device_output).is_ok());
    CHECK(encoder.forward(&context, device_input, &device_output).is_ok());
    CHECK(encoder.workspace_rebuild_count() == 1);
    std::vector<float> actual(expected.values.size());
    CHECK(context.download(
        device_output, actual.data(), actual.size() * sizeof(float)).is_ok());
    CHECK(context.synchronize().is_ok());
    for (std::size_t index = 0; index < actual.size(); ++index) {
        CHECK(std::fabs(actual[index] - expected.values[index]) <= 3.0e-4F);
    }

    const multimodal::FloatMatrix second_input{
        tokens, hidden,
        {-0.8F, 0.1F, 0.3F, -0.5F,
         0.6F, -0.7F, 0.2F, 0.4F,
         -0.1F, 0.9F, -0.6F, 0.5F}};
    multimodal::FloatMatrix second_expected;
    CHECK(multimodal::run_siglip_encoder(
        second_input, blocks.size(), heads, epsilon, provider, post_norm,
        &second_expected, &summaries, &stats, &error));
    std::vector<float> packed_input = input.values;
    packed_input.insert(
        packed_input.end(), second_input.values.begin(), second_input.values.end());
    std::vector<float> packed_expected = expected.values;
    packed_expected.insert(
        packed_expected.end(), second_expected.values.begin(),
        second_expected.values.end());
    CudaTensor device_batch_input;
    CudaTensor device_batch_output;
    CHECK(CudaTensor::create_contiguous(
        {2 * tokens, hidden}, DType::Float32, &device_batch_input).is_ok());
    CHECK(CudaTensor::create_contiguous(
        {2 * tokens, hidden}, DType::Float32, &device_batch_output).is_ok());
    CHECK(context.upload(
        &device_batch_input, packed_input.data(),
        packed_input.size() * sizeof(float)).is_ok());
    CHECK(encoder.forward_batch(
        &context, device_batch_input, 2, &device_batch_output).is_ok());
    CHECK(encoder.workspace_rebuild_count() == 2);
    actual.resize(packed_expected.size());
    CHECK(context.download(
        device_batch_output, actual.data(),
        actual.size() * sizeof(float)).is_ok());
    CHECK(context.synchronize().is_ok());
    for (std::size_t index = 0; index < actual.size(); ++index) {
        CHECK(std::fabs(actual[index] - packed_expected[index]) <= 3.0e-4F);
    }
    CHECK(!encoder.forward_batch(
        &context, device_batch_input, 4, &device_batch_output).is_ok());
    std::cout << "multimodal_cuda_vision_encoder_test: PASS\n";
    return 0;
}

#undef CHECK
