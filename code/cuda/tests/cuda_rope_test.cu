// CUDA Backend：验证 RoPE 原地旋转的数值、边界、ExecutionContext 和设备计时。
// 职责：使用 TinyLlama 同构 CPU 公式检查多 token/head 与局部 rotary dimension。
// 通过条件：CPU/CUDA 逐元素误差通过，错误请求被拒绝并输出固定 workload event 均值。
#include "cpp_llm_engine/core/rope_contract.h"
#include "cpp_llm_engine/core/tensor_contract.h"
#include "cpp_llm_engine/cuda/cuda_event.h"
#include "cpp_llm_engine/cuda/cuda_execution_context.h"
#include "cpp_llm_engine/cuda/cuda_stream.h"
#include "cpp_llm_engine/cuda/cuda_tensor.h"
#include "cpp_llm_engine/cuda/device_buffer.h"
#include "cpp_llm_engine/cuda/rope_backend.h"

#include <cuda_runtime_api.h>

#include <cmath>
#include <cstddef>
#include <iostream>
#include <limits>
#include <vector>

#define CHECK(condition)                                                        \
    do {                                                                        \
        if (!(condition)) {                                                     \
            std::cerr << "check failed: " #condition << " at line " << __LINE__ \
                      << '\n';                                                  \
            return false;                                                       \
        }                                                                       \
    } while (false)

namespace {

using cpp_llm_engine::core::DType;
using cpp_llm_engine::core::RotaryPairing;
using cpp_llm_engine::core::RotaryEmbeddingRequest;
using cpp_llm_engine::cuda::CudaEvent;
using cpp_llm_engine::cuda::CudaExecutionContext;
using cpp_llm_engine::cuda::CudaStream;
using cpp_llm_engine::cuda::CudaTensor;
using cpp_llm_engine::cuda::DeviceBuffer;

void apply_cpu_rope(
    std::vector<float>* values,
    std::size_t tokens,
    std::size_t heads,
    std::size_t head_dimension,
    std::size_t rotary_dimension,
    std::size_t position_offset,
    float frequency_base,
    RotaryPairing pairing) {
    for (std::size_t token = 0; token < tokens; ++token) {
        for (std::size_t head = 0; head < heads; ++head) {
            float* vector = values->data() +
                            (token * heads + head) * head_dimension;
            for (std::size_t pair = 0; pair < rotary_dimension / 2; ++pair) {
                const float exponent =
                    -2.0F * static_cast<float>(pair) /
                    static_cast<float>(rotary_dimension);
                const float angle = static_cast<float>(position_offset + token) *
                                    std::pow(frequency_base, exponent);
                const float cosine = std::cos(angle);
                const float sine = std::sin(angle);
                const std::size_t first = pairing == RotaryPairing::SplitHalf
                    ? pair : pair * 2;
                const std::size_t second = pairing == RotaryPairing::SplitHalf
                    ? pair + rotary_dimension / 2 : pair * 2 + 1;
                const float x = vector[first];
                const float y = vector[second];
                vector[first] = x * cosine - y * sine;
                vector[second] = x * sine + y * cosine;
            }
        }
    }
}

void apply_cpu_rope_positions(
    std::vector<float>* values,
    const std::vector<std::size_t>& positions,
    std::size_t heads,
    std::size_t head_dimension,
    std::size_t rotary_dimension,
    float frequency_base,
    RotaryPairing pairing) {
    for (std::size_t row = 0; row < positions.size(); ++row) {
        for (std::size_t head = 0; head < heads; ++head) {
            float* vector = values->data() +
                (row * heads + head) * head_dimension;
            for (std::size_t pair = 0; pair < rotary_dimension / 2; ++pair) {
                const float exponent =
                    -2.0F * static_cast<float>(pair) /
                    static_cast<float>(rotary_dimension);
                const float angle = static_cast<float>(positions[row]) *
                    std::pow(frequency_base, exponent);
                const float cosine = std::cos(angle);
                const float sine = std::sin(angle);
                const std::size_t first = pairing == RotaryPairing::SplitHalf
                    ? pair : pair * 2;
                const std::size_t second = pairing == RotaryPairing::SplitHalf
                    ? pair + rotary_dimension / 2 : pair * 2 + 1;
                const float x = vector[first];
                const float y = vector[second];
                vector[first] = x * cosine - y * sine;
                vector[second] = x * sine + y * cosine;
            }
        }
    }
}

bool run_correctness_case(
    std::size_t tokens,
    std::size_t heads,
    std::size_t head_dimension,
    std::size_t rotary_dimension,
    std::size_t position_offset,
    RotaryPairing pairing = RotaryPairing::Adjacent) {
    const std::size_t elements = tokens * heads * head_dimension;
    std::vector<float> input(elements);
    for (std::size_t index = 0; index < elements; ++index) {
        input[index] = static_cast<float>(static_cast<int>(index % 29) - 14) * 0.125F;
    }
    std::vector<float> golden = input;
    apply_cpu_rope(
        &golden, tokens, heads, head_dimension, rotary_dimension,
        position_offset, 10000.0F, pairing);

    CudaExecutionContext context;
    CudaTensor tensor;
    CHECK(CudaExecutionContext::create(&context).is_ok());
    CHECK(CudaTensor::create_contiguous(
        {tokens, heads, head_dimension}, DType::Float32, &tensor).is_ok());
    CHECK(context.upload(&tensor, input.data(), input.size() * sizeof(float)).is_ok());
    CHECK(context.rope(
        &tensor, position_offset, rotary_dimension, 10000.0F, pairing).is_ok());
    std::vector<float> output(elements);
    CHECK(context.download(
        tensor, output.data(), output.size() * sizeof(float)).is_ok());
    CHECK(context.synchronize().is_ok());
    for (std::size_t index = 0; index < elements; ++index) {
        CHECK(std::fabs(output[index] - golden[index]) <= 2.0e-5F);
    }
    return true;
}

bool run_batch_position_case(
    RotaryPairing pairing = RotaryPairing::Adjacent) {
    constexpr std::size_t heads = 3;
    constexpr std::size_t head_dimension = 10;
    constexpr std::size_t rotary_dimension = 8;
    const std::vector<std::size_t> positions{0, 7, 31};
    const std::size_t elements = positions.size() * heads * head_dimension;
    std::vector<float> input(elements);
    for (std::size_t index = 0; index < elements; ++index) {
        input[index] = static_cast<float>(static_cast<int>(index % 31) - 15) * 0.0625F;
    }
    std::vector<float> golden = input;
    apply_cpu_rope_positions(
        &golden, positions, heads, head_dimension, rotary_dimension, 10000.0F,
        pairing);

    CudaExecutionContext context;
    CudaTensor tensor;
    CHECK(CudaExecutionContext::create(&context).is_ok());
    CHECK(CudaTensor::create_contiguous(
        {positions.size(), heads, head_dimension}, DType::Float32, &tensor).is_ok());
    CHECK(context.upload(&tensor, input.data(), input.size() * sizeof(float)).is_ok());
    CHECK(context.rope_batch(
        &tensor, positions, rotary_dimension, 10000.0F, pairing).is_ok());
    std::vector<float> output(elements);
    CHECK(context.download(
        tensor, output.data(), output.size() * sizeof(float)).is_ok());
    CHECK(context.synchronize().is_ok());
    for (std::size_t index = 0; index < elements; ++index) {
        CHECK(std::fabs(output[index] - golden[index]) <= 2.0e-5F);
    }

    CHECK(!context.rope_batch(
        &tensor, {0, 1}, rotary_dimension, 10000.0F).is_ok());
    CHECK(!context.rope_batch(
        nullptr, positions, rotary_dimension, 10000.0F).is_ok());
    CHECK(!context.rope_batch(
        &tensor, {}, rotary_dimension, 10000.0F).is_ok());
    return true;
}

bool run_benchmark() {
    constexpr std::size_t tokens = 257;
    constexpr std::size_t heads = 32;
    constexpr std::size_t head_dimension = 64;
    constexpr std::size_t rotary_dimension = 64;
    constexpr int warmup = 1;
    constexpr int iterations = 10;
    const std::size_t elements = tokens * heads * head_dimension;
    const std::size_t bytes = elements * sizeof(float);
    std::vector<float> input(elements, 0.25F);
    DeviceBuffer values;
    CudaStream stream;
    CudaEvent start;
    CudaEvent end;
    CHECK(DeviceBuffer::allocate(bytes, &values).is_ok());
    CHECK(CudaStream::create(&stream).is_ok());
    CHECK(CudaEvent::create(&start).is_ok());
    CHECK(CudaEvent::create(&end).is_ok());
    CHECK(values.copy_from_host_async(input.data(), bytes, stream).is_ok());
    const RotaryEmbeddingRequest request{
        static_cast<float*>(values.data()),
        tokens,
        heads,
        head_dimension,
        rotary_dimension,
        17,
        10000.0F};
    for (int iteration = 0; iteration < warmup; ++iteration) {
        CHECK(cpp_llm_engine::cuda::cuda_rope_backend().apply(request, stream).is_ok());
    }
    CHECK(stream.synchronize().is_ok());
    CHECK(start.record(stream).is_ok());
    for (int iteration = 0; iteration < iterations; ++iteration) {
        CHECK(cpp_llm_engine::cuda::cuda_rope_backend().apply(request, stream).is_ok());
    }
    CHECK(end.record(stream).is_ok());
    CHECK(end.synchronize().is_ok());
    float elapsed_ms = 0.0F;
    CHECK(CudaEvent::elapsed_time_ms(start, end, &elapsed_ms).is_ok());
    CHECK(elapsed_ms >= 0.0F);
    std::cout << "rope_tokens=" << tokens
              << " heads=" << heads
              << " head_dimension=" << head_dimension
              << " warmup=" << warmup
              << " iterations=" << iterations
              << " kernel_mean_ms=" << elapsed_ms / iterations << '\n';
    return true;
}

bool run_error_cases() {
    const auto& backend = cpp_llm_engine::cuda::cuda_rope_backend();
    CudaStream invalid_stream;
    CHECK(backend.kind() == cpp_llm_engine::core::BackendKind::CUDA);
    CHECK(backend.name() == "cuda-rope");
    CHECK(backend.apply(RotaryEmbeddingRequest{}, invalid_stream).is_ok());

    CudaStream stream;
    CHECK(CudaStream::create(&stream).is_ok());
    RotaryEmbeddingRequest request;
    request.token_count = 1;
    request.head_count = 1;
    request.head_dimension = 8;
    request.rotary_dimension = 8;
    CHECK(!backend.apply(request, stream).is_ok());
    float* fake_device_pointer = reinterpret_cast<float*>(0x1);
    request.values = fake_device_pointer;
    request.rotary_dimension = 7;
    CHECK(!backend.apply(request, stream).is_ok());
    request.rotary_dimension = 10;
    CHECK(!backend.apply(request, stream).is_ok());
    request.rotary_dimension = 8;
    request.frequency_base = 0.0F;
    CHECK(!backend.apply(request, stream).is_ok());
    request.frequency_base = 10000.0F;
    request.token_count = 2;
    request.position_offset = std::numeric_limits<std::size_t>::max();
    CHECK(!backend.apply(request, stream).is_ok());
    request.token_count = 1;
    request.position_offset = 0;
    CHECK(!backend.apply(request, invalid_stream).is_ok());

    CudaExecutionContext context;
    CudaTensor rank_two;
    CHECK(CudaExecutionContext::create(&context).is_ok());
    CHECK(CudaTensor::create_contiguous({2, 8}, DType::Float32, &rank_two).is_ok());
    CHECK(!context.rope(&rank_two, 0, 8, 10000.0F).is_ok());
    CHECK(!context.rope(nullptr, 0, 8, 10000.0F).is_ok());
    return true;
}

}  // namespace

int main() {
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count <= 0) return 1;
    if (!run_error_cases()) return 1;
    if (!run_correctness_case(1, 1, 8, 8, 0)) return 1;
    if (!run_correctness_case(3, 2, 8, 6, 5)) return 1;
    if (!run_correctness_case(257, 3, 10, 8, 31)) return 1;
    if (!run_correctness_case(
            3, 2, 128, 128, 5, RotaryPairing::SplitHalf)) return 1;
    if (!run_batch_position_case()) return 1;
    if (!run_batch_position_case(RotaryPairing::SplitHalf)) return 1;
    if (!run_benchmark()) return 1;
    return 0;
}

#undef CHECK
