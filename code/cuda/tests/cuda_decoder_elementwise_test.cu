// CUDA Decoder：验证 RMSNorm 与 SwiGLU 的数值和错误路径。
// 职责：用 257/2048 宽度和 Qwen3 [B,H,128] CPU golden 覆盖批量归一化。
// 通过条件：逐元素误差通过，shape、epsilon 和空输出错误被拒绝。
#include "cpp_llm_engine/core/tensor_contract.h"
#include "cpp_llm_engine/cuda/cuda_execution_context.h"
#include "cpp_llm_engine/cuda/cuda_tensor.h"

#include <cuda_runtime_api.h>

#include <cmath>
#include <cstddef>
#include <iostream>
#include <vector>

#define CHECK(condition) do { if (!(condition)) { \
    std::cerr << "check failed: " #condition " at line " << __LINE__ << '\n'; \
    return false; } } while (false)

namespace {

using cpp_llm_engine::core::DType;
using cpp_llm_engine::cuda::CudaExecutionContext;
using cpp_llm_engine::cuda::CudaTensor;

bool run_case(std::size_t elements) {
    CudaExecutionContext context;
    CHECK(CudaExecutionContext::create(&context).is_ok());
    std::vector<float> input(elements);
    std::vector<float> weight(elements);
    std::vector<float> up(elements);
    for (std::size_t index = 0; index < elements; ++index) {
        input[index] = static_cast<float>(static_cast<int>(index % 19) - 9) * 0.07F;
        weight[index] = 0.8F + static_cast<float>(index % 5) * 0.05F;
        up[index] = static_cast<float>(static_cast<int>(index % 13) - 6) * 0.04F;
    }
    CudaTensor input_tensor;
    CudaTensor weight_tensor;
    CudaTensor up_tensor;
    CudaTensor norm_output;
    CudaTensor swiglu_output;
    for (CudaTensor* tensor : {&input_tensor, &weight_tensor, &up_tensor,
                               &norm_output, &swiglu_output}) {
        CHECK(CudaTensor::create_contiguous({elements}, DType::Float32, tensor).is_ok());
    }
    CHECK(context.upload(&input_tensor, input.data(), elements * sizeof(float)).is_ok());
    CHECK(context.upload(&weight_tensor, weight.data(), elements * sizeof(float)).is_ok());
    CHECK(context.upload(&up_tensor, up.data(), elements * sizeof(float)).is_ok());
    constexpr float epsilon = 1.0e-5F;
    CHECK(context.rms_norm(input_tensor, weight_tensor, &norm_output, epsilon).is_ok());
    CHECK(context.swiglu(input_tensor, up_tensor, &swiglu_output).is_ok());
    std::vector<float> norm(elements);
    std::vector<float> swiglu(elements);
    CHECK(context.download(norm_output, norm.data(), elements * sizeof(float)).is_ok());
    CHECK(context.download(swiglu_output, swiglu.data(), elements * sizeof(float)).is_ok());
    CHECK(context.synchronize().is_ok());
    float mean_square = 0.0F;
    for (const float value : input) mean_square += value * value;
    const float inverse = 1.0F /
        std::sqrt(mean_square / static_cast<float>(elements) + epsilon);
    for (std::size_t index = 0; index < elements; ++index) {
        const float expected_norm = input[index] * inverse * weight[index];
        const float expected_swiglu =
            input[index] / (1.0F + std::exp(-input[index])) * up[index];
        CHECK(std::fabs(norm[index] - expected_norm) <= 2.0e-5F);
        CHECK(std::fabs(swiglu[index] - expected_swiglu) <= 2.0e-6F);
    }
    CudaTensor wrong;
    CHECK(CudaTensor::create_contiguous({elements + 1}, DType::Float32, &wrong).is_ok());
    CHECK(!context.rms_norm(input_tensor, wrong, &norm_output, epsilon).is_ok());
    CHECK(!context.rms_norm(input_tensor, weight_tensor, &norm_output, 0.0F).is_ok());
    CHECK(!context.swiglu(input_tensor, up_tensor, nullptr).is_ok());
    return true;
}

bool run_batch_case(const std::vector<std::size_t>& shape) {
    CHECK(shape.size() >= 2);
    const std::size_t width = shape.back();
    std::size_t rows = 1;
    for (std::size_t index = 0; index + 1 < shape.size(); ++index) {
        rows *= shape[index];
    }
    CudaExecutionContext context;
    CHECK(CudaExecutionContext::create(&context).is_ok());
    std::vector<float> input(rows * width);
    std::vector<float> weight(width);
    for (std::size_t col = 0; col < width; ++col) {
        weight[col] = 0.75F + static_cast<float>(col % 7) * 0.03F;
    }
    for (std::size_t index = 0; index < input.size(); ++index) {
        input[index] = static_cast<float>(static_cast<int>(index % 23) - 11) * 0.05F;
    }
    CudaTensor input_tensor;
    CudaTensor weight_tensor;
    CudaTensor output_tensor;
    CHECK(CudaTensor::create_contiguous(
        shape, DType::Float32, &input_tensor).is_ok());
    CHECK(CudaTensor::create_contiguous(
        {width}, DType::Float32, &weight_tensor).is_ok());
    CHECK(CudaTensor::create_contiguous(
        shape, DType::Float32, &output_tensor).is_ok());
    CHECK(context.upload(
        &input_tensor, input.data(), input.size() * sizeof(float)).is_ok());
    CHECK(context.upload(
        &weight_tensor, weight.data(), weight.size() * sizeof(float)).is_ok());
    constexpr float epsilon = 1.0e-5F;
    CHECK(context.rms_norm_batch(
        input_tensor, weight_tensor, &output_tensor, epsilon).is_ok());
    std::vector<float> output(input.size());
    CHECK(context.download(
        output_tensor, output.data(), output.size() * sizeof(float)).is_ok());
    CHECK(context.synchronize().is_ok());
    for (std::size_t row = 0; row < rows; ++row) {
        float mean_square = 0.0F;
        for (std::size_t col = 0; col < width; ++col) {
            const float value = input[row * width + col];
            mean_square += value * value;
        }
        const float inverse = 1.0F /
            std::sqrt(mean_square / static_cast<float>(width) + epsilon);
        for (std::size_t col = 0; col < width; ++col) {
            const float expected = input[row * width + col] * inverse * weight[col];
            CHECK(std::fabs(output[row * width + col] - expected) <= 2.0e-5F);
        }
    }
    CudaTensor wrong_weight;
    CHECK(CudaTensor::create_contiguous(
        {width + 1}, DType::Float32, &wrong_weight).is_ok());
    CHECK(!context.rms_norm_batch(
        input_tensor, wrong_weight, &output_tensor, epsilon).is_ok());
    CHECK(!context.rms_norm_batch(
        weight_tensor, weight_tensor, &output_tensor, epsilon).is_ok());
    return true;
}

}  // namespace

int main() {
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count <= 0) return 1;
    if (!run_case(257)) return 1;
    if (!run_case(2048)) return 1;
    if (!run_batch_case({3, 257})) return 1;
    if (!run_batch_case({4, 2048})) return 1;
    if (!run_batch_case({2, 3, 128})) return 1;
    return 0;
}

#undef CHECK
