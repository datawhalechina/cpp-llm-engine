// CUDA Quantized Linear：验证 Q4_K/Q6_K dequantize + SGEMM 基线。
// 职责：与 Chapter05 CPU reference 对照，并覆盖 payload、shape、移动和 workspace 复用。
// 通过条件：真实 GPU 输出满足误差阈值，且测试不直接管理 DeviceBuffer/stream/handle。
#include "chapter05_quantization/q4k_dequantize.h"
#include "chapter05_quantization/q6k_dequantize.h"
#include "cpp_llm_engine/core/linear_contract.h"
#include "cpp_llm_engine/cuda/cuda_execution_context.h"
#include "cpp_llm_engine/cuda/cuda_quantized_matrix.h"
#include "cpp_llm_engine/cuda/cuda_tensor.h"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <string>
#include <string_view>
#include <utility>
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

std::vector<float> cpu_linear(
    const std::vector<float>& weights,
    std::size_t rows,
    std::size_t cols,
    const std::vector<float>& input) {
    std::vector<float> output(rows, 0.0F);
    for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t col = 0; col < cols; ++col) {
            output[row] += weights[row * cols + col] * input[col];
        }
    }
    return output;
}

bool close_enough(float actual, float expected) {
    const float tolerance = 1.0e-3F * std::max(1.0F, std::abs(expected));
    return std::abs(actual - expected) <= tolerance;
}

std::vector<std::uint8_t> make_q4k_payload() {
    using chapter05_quantization::kQ4KBlockBytes;
    std::vector<std::uint8_t> payload(2 * kQ4KBlockBytes, 0);

    payload[0] = 0x00;
    payload[1] = 0x3c;  // fp16 1.0
    for (std::size_t index = 4; index < 12; ++index) payload[index] = 1;
    for (std::size_t index = 12; index < kQ4KBlockBytes; ++index) {
        payload[index] = 0x11;
    }

    const std::size_t offset = kQ4KBlockBytes;
    payload[offset + 0] = 0x00;
    payload[offset + 1] = 0x38;  // fp16 0.5
    payload[offset + 2] = 0x00;
    payload[offset + 3] = 0x34;  // fp16 0.25
    for (std::size_t index = 0; index < 12; ++index) {
        payload[offset + 4 + index] = static_cast<std::uint8_t>(3 + index * 7);
    }
    for (std::size_t index = 0; index < 128; ++index) {
        payload[offset + 16 + index] =
            static_cast<std::uint8_t>((index * 29 + 11) & 0xff);
    }
    return payload;
}

std::vector<std::uint8_t> make_q6k_payload() {
    using chapter05_quantization::kQ6KBlockBytes;
    std::vector<std::uint8_t> payload(2 * kQ6KBlockBytes, 0);
    for (std::size_t block = 0; block < 2; ++block) {
        const std::size_t offset = block * kQ6KBlockBytes;
        payload[offset + 0] = 0x21;
        payload[offset + 32] = 0x43;
        payload[offset + 128] = 0x39;
        for (std::size_t index = 192; index < 208; ++index) {
            payload[offset + index] = static_cast<std::uint8_t>(1 + block);
        }
        payload[offset + 208] = 0x00;
        payload[offset + 209] = block == 0 ? 0x3c : 0x38;  // fp16 1.0 / 0.5
    }
    return payload;
}

}  // namespace

int main() {
    using cpp_llm_engine::core::DType;
    using cpp_llm_engine::core::WeightFormat;
    using cpp_llm_engine::cuda::CudaExecutionContext;
    using cpp_llm_engine::cuda::CudaQuantizedMatrix;
    using cpp_llm_engine::cuda::CudaTensor;

    int device_count = 0;
    CHECK(cudaGetDeviceCount(&device_count) == cudaSuccess);
    CHECK(device_count > 0);

    CudaQuantizedMatrix invalid;
    CHECK(!CudaQuantizedMatrix::create(1, 256, WeightFormat::Q4K, nullptr).is_ok());
    CHECK(!CudaQuantizedMatrix::create(1, 255, WeightFormat::Q4K, &invalid).is_ok());
    CHECK(!CudaQuantizedMatrix::create(1, 256, WeightFormat::Float32, &invalid).is_ok());
    CHECK(!CudaQuantizedMatrix::create(
        std::numeric_limits<std::size_t>::max(),
        256,
        WeightFormat::Q6K,
        &invalid).is_ok());

    CudaExecutionContext context;
    CHECK(CudaExecutionContext::create(&context).is_ok());
    std::vector<float> input_host(256, 0.0F);
    for (std::size_t index = 0; index < input_host.size(); ++index) {
        input_host[index] =
            static_cast<float>(static_cast<int>(index % 17) - 8) / 7.0F;
    }
    CudaTensor input;
    CudaTensor output;
    CHECK(CudaTensor::create_contiguous({256}, DType::Float32, &input).is_ok());
    CHECK(CudaTensor::create_contiguous({2}, DType::Float32, &output).is_ok());
    CHECK(context.upload(&input, input_host.data(), input_host.size() * sizeof(float)).is_ok());

    const std::vector<std::uint8_t> q4k_payload = make_q4k_payload();
    std::vector<float> q4k_decoded;
    std::string error;
    CHECK(chapter05_quantization::dequantize_q4_k(
        q4k_payload.data(), q4k_payload.size(), 512, &q4k_decoded, &error));
    const std::vector<float> q4k_expected =
        cpu_linear(q4k_decoded, 2, 256, input_host);
    CudaQuantizedMatrix q4k_weights;
    CHECK(CudaQuantizedMatrix::create(2, 256, WeightFormat::Q4K, &q4k_weights).is_ok());
    CHECK(q4k_weights.size_bytes() == q4k_payload.size());
    CHECK(!context.upload(
        &q4k_weights, q4k_payload.data(), q4k_payload.size() - 1).is_ok());
    CHECK(context.upload(&q4k_weights, q4k_payload.data(), q4k_payload.size()).is_ok());
    CHECK(context.quantized_linear(q4k_weights, input, &output).is_ok());
    std::vector<float> output_host(2, 0.0F);
    CHECK(context.download(output, output_host.data(), output_host.size() * sizeof(float)).is_ok());
    CHECK(context.synchronize().is_ok());
    CHECK(close_enough(output_host[0], q4k_expected[0]));
    CHECK(close_enough(output_host[1], q4k_expected[1]));
    CudaTensor rank_three_output;
    CHECK(CudaTensor::create_contiguous(
        {1, 1, 2}, DType::Float32, &rank_three_output).is_ok());
    CHECK(context.linear(
        input,
        cpp_llm_engine::cuda::CudaLinearWeight::quantized(&q4k_weights),
        &rank_three_output).is_ok());
    CHECK(context.download(
        rank_three_output, output_host.data(), output_host.size() * sizeof(float)).is_ok());
    CHECK(context.synchronize().is_ok());
    CHECK(close_enough(output_host[0], q4k_expected[0]));
    CHECK(close_enough(output_host[1], q4k_expected[1]));
    CHECK(context.workspace_capacity_bytes() == 512 * sizeof(float));
    CHECK(context.workspace_allocation_count() == 1);

    const std::vector<std::uint8_t> q6k_payload = make_q6k_payload();
    std::vector<float> q6k_decoded;
    CHECK(chapter05_quantization::dequantize_q6_k(
        q6k_payload.data(), q6k_payload.size(), 512, &q6k_decoded, &error));
    const std::vector<float> q6k_expected =
        cpu_linear(q6k_decoded, 2, 256, input_host);
    CudaQuantizedMatrix q6k_source;
    CHECK(CudaQuantizedMatrix::create(2, 256, WeightFormat::Q6K, &q6k_source).is_ok());
    CudaQuantizedMatrix q6k_weights(std::move(q6k_source));
    CHECK(q6k_source.empty());
    CHECK(q6k_source.rows() == 0);
    CHECK(q6k_weights.format() == WeightFormat::Q6K);
    CHECK(context.upload(&q6k_weights, q6k_payload.data(), q6k_payload.size()).is_ok());
    CHECK(context.quantized_linear(q6k_weights, input, &output).is_ok());
    CHECK(context.download(output, output_host.data(), output_host.size() * sizeof(float)).is_ok());
    CHECK(context.synchronize().is_ok());
    CHECK(close_enough(output_host[0], q6k_expected[0]));
    CHECK(close_enough(output_host[1], q6k_expected[1]));
    CHECK(context.workspace_capacity_bytes() == 512 * sizeof(float));
    CHECK(context.workspace_allocation_count() == 1);

    CudaTensor wrong_input;
    CudaTensor wrong_output;
    CHECK(CudaTensor::create_contiguous({257}, DType::Float32, &wrong_input).is_ok());
    CHECK(CudaTensor::create_contiguous({3}, DType::Float32, &wrong_output).is_ok());
    CHECK(!context.quantized_linear(q4k_weights, wrong_input, &output).is_ok());
    CHECK(!context.quantized_linear(q4k_weights, input, &wrong_output).is_ok());
    CHECK(!context.quantized_linear(q4k_weights, input, nullptr).is_ok());

    constexpr std::size_t benchmark_rows = 256;
    constexpr int warmup = 1;
    constexpr int iterations = 10;
    std::vector<std::uint8_t> q4k_benchmark_payload(
        benchmark_rows * chapter05_quantization::kQ4KBlockBytes);
    std::vector<std::uint8_t> q6k_benchmark_payload(
        benchmark_rows * chapter05_quantization::kQ6KBlockBytes);
    for (std::size_t row = 0; row < benchmark_rows; ++row) {
        std::copy_n(
            q4k_payload.data(),
            chapter05_quantization::kQ4KBlockBytes,
            q4k_benchmark_payload.data() +
                row * chapter05_quantization::kQ4KBlockBytes);
        std::copy_n(
            q6k_payload.data(),
            chapter05_quantization::kQ6KBlockBytes,
            q6k_benchmark_payload.data() +
                row * chapter05_quantization::kQ6KBlockBytes);
    }
    CudaQuantizedMatrix q4k_benchmark_weights;
    CudaQuantizedMatrix q6k_benchmark_weights;
    CudaTensor benchmark_output;
    CHECK(CudaQuantizedMatrix::create(
        benchmark_rows, 256, WeightFormat::Q4K, &q4k_benchmark_weights).is_ok());
    CHECK(CudaQuantizedMatrix::create(
        benchmark_rows, 256, WeightFormat::Q6K, &q6k_benchmark_weights).is_ok());
    CHECK(CudaTensor::create_contiguous(
        {benchmark_rows}, DType::Float32, &benchmark_output).is_ok());
    CHECK(context.upload(
        &q4k_benchmark_weights,
        q4k_benchmark_payload.data(),
        q4k_benchmark_payload.size()).is_ok());
    CHECK(context.upload(
        &q6k_benchmark_weights,
        q6k_benchmark_payload.data(),
        q6k_benchmark_payload.size()).is_ok());

    const auto run_benchmark = [&](const CudaQuantizedMatrix& weights,
                                   std::string_view format) {
        for (int iteration = 0; iteration < warmup; ++iteration) {
            if (!context.quantized_linear(weights, input, &benchmark_output).is_ok()) {
                return false;
            }
        }
        if (!context.synchronize().is_ok()) return false;
        const auto start = std::chrono::steady_clock::now();
        for (int iteration = 0; iteration < iterations; ++iteration) {
            if (!context.quantized_linear(weights, input, &benchmark_output).is_ok()) {
                return false;
            }
        }
        if (!context.synchronize().is_ok()) return false;
        const auto end = std::chrono::steady_clock::now();
        const double elapsed_ms =
            std::chrono::duration<double, std::milli>(end - start).count();
        std::cout << "quantized_format=" << format
                  << " rows=" << benchmark_rows
                  << " cols=256 warmup=" << warmup
                  << " iterations=" << iterations
                  << " dequant_sgemm_host_mean_ms=" << elapsed_ms / iterations
                  << '\n';
        return true;
    };
    CHECK(run_benchmark(q4k_benchmark_weights, "Q4_K"));
    CHECK(run_benchmark(q6k_benchmark_weights, "Q6_K"));
    CHECK(context.workspace_capacity_bytes() == benchmark_rows * 256 * sizeof(float));
    CHECK(context.workspace_allocation_count() == 2);
    std::vector<float> benchmark_output_host(benchmark_rows, 0.0F);
    CHECK(context.download(
        benchmark_output,
        benchmark_output_host.data(),
        benchmark_output_host.size() * sizeof(float)).is_ok());
    CHECK(context.synchronize().is_ok());
    CHECK(std::all_of(
        benchmark_output_host.begin(),
        benchmark_output_host.end(),
        [](float value) { return std::isfinite(value); }));
    return 0;
}

#undef CHECK
