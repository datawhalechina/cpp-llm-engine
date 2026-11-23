// 多模态 CUDA：验证单个 SigLIP Block 与 CPU reference 一致。
#include "multimodal/cuda_siglip_block.h"

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

multimodal::LinearWeights patterned_linear(
    std::size_t input_size,
    std::size_t output_size,
    float scale) {
    multimodal::LinearWeights result;
    result.input_size = input_size;
    result.output_size = output_size;
    result.weight.resize(input_size * output_size);
    result.bias.resize(output_size);
    for (std::size_t output = 0; output < output_size; ++output) {
        result.bias[output] = scale * static_cast<float>(output + 1) * 0.1F;
        for (std::size_t input = 0; input < input_size; ++input) {
            const int pattern = static_cast<int>((output * 7 + input * 3) % 11) - 5;
            result.weight[output * input_size + input] =
                scale * static_cast<float>(pattern);
        }
    }
    return result;
}

}  // namespace

int main() {
    using cpp_llm_engine::core::DType;
    using cpp_llm_engine::cuda::CudaExecutionContext;
    using cpp_llm_engine::cuda::CudaTensor;

    constexpr std::size_t tokens = 3;
    constexpr std::size_t hidden = 4;
    constexpr std::size_t heads = 2;
    constexpr std::size_t feed_forward = 6;
    constexpr float epsilon = 1.0e-5F;
    const multimodal::FloatMatrix input{
        tokens, hidden,
        {0.2F, -0.4F, 0.6F, 0.8F,
         -0.3F, 0.7F, -0.5F, 0.1F,
         0.9F, -0.2F, 0.4F, -0.8F}};
    multimodal::SiglipBlockWeights weights;
    weights.layer_norm_1 = {
        {1.1F, 0.9F, 1.2F, 0.8F}, {0.01F, -0.02F, 0.03F, -0.04F}};
    weights.query = patterned_linear(hidden, hidden, 0.05F);
    weights.key = patterned_linear(hidden, hidden, -0.04F);
    weights.value = patterned_linear(hidden, hidden, 0.03F);
    weights.attention_output = patterned_linear(hidden, hidden, 0.02F);
    weights.layer_norm_2 = {
        {0.8F, 1.2F, 0.9F, 1.1F}, {-0.01F, 0.02F, -0.03F, 0.04F}};
    weights.feed_forward_input =
        patterned_linear(hidden, feed_forward, 0.025F);
    weights.feed_forward_output =
        patterned_linear(feed_forward, hidden, -0.02F);

    multimodal::SiglipBlockTrace expected;
    std::string error;
    CHECK(multimodal::run_siglip_block(
        input, heads, epsilon, weights, &expected, &error));

    CudaExecutionContext context;
    multimodal::CudaSiglipBlock block;
    CudaTensor device_input;
    CudaTensor output;
    CHECK(CudaExecutionContext::create(&context).is_ok());
    CHECK(CudaTensor::create_contiguous(
        {tokens, hidden}, DType::Float32, &device_input).is_ok());
    CHECK(CudaTensor::create_contiguous(
        {tokens, hidden}, DType::Float32, &output).is_ok());
    CHECK(context.upload(
        &device_input, input.values.data(),
        input.values.size() * sizeof(float)).is_ok());
    CHECK(multimodal::CudaSiglipBlock::upload(
        &context, weights, heads, epsilon, &block).is_ok());
    const auto forward_status = block.forward(&context, device_input, &output);
    if (!forward_status.is_ok()) {
        std::cerr << "SigLIP CUDA forward failed: "
                  << forward_status.message() << '\n';
        return 1;
    }

    std::vector<float> actual(expected.output.values.size());
    CHECK(context.download(
        output, actual.data(), actual.size() * sizeof(float)).is_ok());
    CHECK(context.synchronize().is_ok());
    for (std::size_t index = 0; index < actual.size(); ++index) {
        CHECK(std::fabs(actual[index] - expected.output.values[index]) <=
              2.0e-4F);
    }

    multimodal::CudaSiglipBlockWorkspace workspace;
    CHECK(multimodal::CudaSiglipBlockWorkspace::create(
        tokens, hidden, heads, feed_forward, &workspace).is_ok());
    CHECK(workspace.token_count() == tokens);
    CHECK(block.forward_with_workspace(
        &context, device_input, &workspace, &output).is_ok());
    CHECK(block.forward_with_workspace(
        &context, device_input, &workspace, &output).is_ok());
    CHECK(context.download(
        output, actual.data(), actual.size() * sizeof(float)).is_ok());
    CHECK(context.synchronize().is_ok());
    for (std::size_t index = 0; index < actual.size(); ++index) {
        CHECK(std::fabs(actual[index] - expected.output.values[index]) <=
              2.0e-4F);
    }

    multimodal::CudaSiglipBlock invalid;
    CHECK(!multimodal::CudaSiglipBlock::upload(
        &context, weights, 3, epsilon, &invalid).is_ok());
    CudaTensor wrong_output;
    CHECK(CudaTensor::create_contiguous(
        {tokens, hidden + 1}, DType::Float32, &wrong_output).is_ok());
    CHECK(!block.forward(&context, device_input, &wrong_output).is_ok());
    multimodal::CudaSiglipBlockWorkspace wrong_workspace;
    CHECK(multimodal::CudaSiglipBlockWorkspace::create(
        tokens + 1, hidden, heads, feed_forward, &wrong_workspace).is_ok());
    CHECK(!block.forward_with_workspace(
        &context, device_input, &wrong_workspace, &output).is_ok());

    std::cout << "multimodal_cuda_siglip_block_test: PASS\n";
    return 0;
}
