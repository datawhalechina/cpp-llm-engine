// 多模态 CUDA：验证空间合并与 IDEFICS3 Projector 和 CPU reference 一致。
#include "multimodal/cuda_vision_projector.h"

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

int main() {
    using cpp_llm_engine::core::DType;
    using cpp_llm_engine::cuda::CudaExecutionContext;
    using cpp_llm_engine::cuda::CudaTensor;

    constexpr std::size_t grid_width = 4;
    constexpr std::size_t grid_height = 2;
    constexpr std::size_t hidden_size = 2;
    constexpr std::size_t merge_size = 2;
    constexpr std::size_t merged_width =
        merge_size * merge_size * hidden_size;
    constexpr std::size_t output_size = 3;

    multimodal::FloatMatrix input{
        grid_width * grid_height, hidden_size,
        {0.0F, 1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F,
         8.0F, 9.0F, 10.0F, 11.0F, 12.0F, 13.0F, 14.0F, 15.0F}};
    multimodal::LinearWeights weights{
        merged_width, output_size,
        {0.1F, 0.2F, 0.3F, 0.4F, 0.5F, 0.6F, 0.7F, 0.8F,
         -0.2F, 0.1F, -0.4F, 0.3F, -0.6F, 0.5F, -0.8F, 0.7F,
         1.0F, -1.0F, 0.5F, -0.5F, 0.25F, -0.25F, 0.125F, -0.125F},
        {}};

    multimodal::Idefics3ProjectorTrace expected;
    std::string error;
    CHECK(multimodal::run_idefics3_projector(
        input, grid_width, grid_height, merge_size, weights, &expected,
        &error));

    CudaExecutionContext context;
    multimodal::CudaIdefics3Projector projector;
    CudaTensor device_input;
    CudaTensor output;
    CHECK(CudaExecutionContext::create(&context).is_ok());
    CHECK(CudaTensor::create_contiguous(
        {input.rows, input.cols}, DType::Float32, &device_input).is_ok());
    CHECK(context.upload(
        &device_input, input.values.data(),
        input.values.size() * sizeof(float)).is_ok());
    CHECK(multimodal::CudaIdefics3Projector::upload(
        &context, weights, &projector).is_ok());
    CHECK(projector.valid());
    CHECK(CudaTensor::create_contiguous(
        {expected.output.rows, expected.output.cols}, DType::Float32,
        &output).is_ok());
    CHECK(projector.forward(
        &context, device_input, grid_width, grid_height, merge_size,
        &output).is_ok());

    std::vector<float> actual(expected.output.values.size());
    CHECK(context.download(
        output, actual.data(), actual.size() * sizeof(float)).is_ok());
    CHECK(context.synchronize().is_ok());
    for (std::size_t index = 0; index < actual.size(); ++index) {
        CHECK(std::fabs(actual[index] - expected.output.values[index]) <=
              1.0e-5F);
    }

    CHECK(!projector.forward(
        &context, device_input, grid_width, grid_height, 3, &output).is_ok());
    CudaTensor wrong_output;
    CHECK(CudaTensor::create_contiguous(
        {expected.output.rows, output_size + 1}, DType::Float32,
        &wrong_output).is_ok());
    CHECK(!projector.forward(
        &context, device_input, grid_width, grid_height, merge_size,
        &wrong_output).is_ok());

    std::cout << "multimodal_cuda_vision_projector_test: PASS\n";
    return 0;
}
