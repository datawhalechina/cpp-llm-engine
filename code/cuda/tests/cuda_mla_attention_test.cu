// CUDA Backend：验证压缩 MLA 的双词元数值、缓存增长和请求隔离。
#include "cpp_llm_engine/core/tensor_contract.h"
#include "cpp_llm_engine/cuda/cuda_mla_cache.h"
#include "cpp_llm_engine/cuda/cuda_stream.h"
#include "cpp_llm_engine/cuda/cuda_tensor.h"
#include "cpp_llm_engine/cuda/mla_attention_backend.h"

#include <cuda_runtime_api.h>

#include <algorithm>
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
using cpp_llm_engine::cuda::CudaMlaAttentionRequest;
using cpp_llm_engine::cuda::CudaMlaCache;
using cpp_llm_engine::cuda::CudaStream;
using cpp_llm_engine::cuda::CudaTensor;

float dot(const float* left, const float* right, std::size_t count) {
    float result = 0.0F;
    for (std::size_t index = 0; index < count; ++index) {
        result += left[index] * right[index];
    }
    return result;
}

std::vector<float> cpu_mla(
    const std::vector<float>& query_nope,
    const std::vector<float>& query_rope,
    const std::vector<float>& kv_b,
    const std::vector<float>& latent_cache,
    const std::vector<float>& rope_cache,
    std::size_t tokens,
    std::size_t heads,
    std::size_t nope,
    std::size_t rope,
    std::size_t value,
    std::size_t latent,
    float scale) {
    std::vector<float> output(heads * value, 0.0F);
    std::vector<float> scores(tokens, 0.0F);
    for (std::size_t head = 0; head < heads; ++head) {
        const std::size_t matrix_offset = head * (nope + value) * latent;
        float maximum = -std::numeric_limits<float>::infinity();
        for (std::size_t token = 0; token < tokens; ++token) {
            const float* latent_row = latent_cache.data() + token * latent;
            float score = dot(
                query_rope.data() + head * rope,
                rope_cache.data() + token * rope, rope);
            for (std::size_t dim = 0; dim < nope; ++dim) {
                score += query_nope[head * nope + dim] * dot(
                    kv_b.data() + matrix_offset + dim * latent,
                    latent_row, latent);
            }
            scores[token] = score * scale;
            maximum = std::max(maximum, scores[token]);
        }
        float denominator = 0.0F;
        for (float& score : scores) {
            score = std::exp(score - maximum);
            denominator += score;
        }
        for (std::size_t token = 0; token < tokens; ++token) {
            const float probability = scores[token] / denominator;
            for (std::size_t dim = 0; dim < value; ++dim) {
                output[head * value + dim] += probability * dot(
                    kv_b.data() + matrix_offset + (nope + dim) * latent,
                    latent_cache.data() + token * latent, latent);
            }
        }
    }
    return output;
}

bool upload(
    CudaTensor* tensor,
    const std::vector<float>& values,
    const CudaStream& stream) {
    return tensor->copy_from_host_async(
        values.data(), values.size() * sizeof(float), stream).is_ok();
}

bool run_mla_case() {
    constexpr std::size_t heads = 2;
    constexpr std::size_t nope = 3;
    constexpr std::size_t rope = 2;
    constexpr std::size_t value = 2;
    constexpr std::size_t latent = 4;
    constexpr float scale = 0.5F;
    const std::vector<float> query_nope{
        0.3F, -0.2F, 0.5F, -0.4F, 0.1F, 0.6F};
    const std::vector<float> query_rope{0.2F, 0.7F, -0.3F, 0.4F};
    std::vector<float> kv_b(heads * (nope + value) * latent);
    for (std::size_t index = 0; index < kv_b.size(); ++index) {
        kv_b[index] = static_cast<float>(static_cast<int>(index % 11) - 5) *
            0.07F;
    }
    const std::vector<float> latent_first{0.4F, -0.1F, 0.2F, 0.8F};
    const std::vector<float> rope_first{0.5F, -0.25F};
    const std::vector<float> latent_second{-0.3F, 0.6F, 0.1F, -0.2F};
    const std::vector<float> rope_second{-0.4F, 0.75F};

    CudaStream stream;
    CudaMlaCache cache;
    CudaMlaCache isolated;
    CudaTensor query_nope_device;
    CudaTensor query_rope_device;
    CudaTensor kv_b_device;
    CudaTensor output_device;
    CudaTensor probabilities_device;
    CudaTensor latent_device;
    CudaTensor rope_device;
    CHECK(CudaStream::create(&stream).is_ok());
    CHECK(CudaMlaCache::create(2, latent, rope, &cache).is_ok());
    CHECK(CudaMlaCache::create(2, latent, rope, &isolated).is_ok());
    CHECK(CudaTensor::create_contiguous(
        {heads, nope}, DType::Float32, &query_nope_device).is_ok());
    CHECK(CudaTensor::create_contiguous(
        {heads, rope}, DType::Float32, &query_rope_device).is_ok());
    CHECK(CudaTensor::create_contiguous(
        {heads * (nope + value), latent}, DType::Float32, &kv_b_device).is_ok());
    CHECK(CudaTensor::create_contiguous(
        {heads, value}, DType::Float32, &output_device).is_ok());
    CHECK(CudaTensor::create_contiguous(
        {heads, 2}, DType::Float32, &probabilities_device).is_ok());
    CHECK(CudaTensor::create_contiguous(
        {latent}, DType::Float32, &latent_device).is_ok());
    CHECK(CudaTensor::create_contiguous(
        {rope}, DType::Float32, &rope_device).is_ok());
    CHECK(upload(&query_nope_device, query_nope, stream));
    CHECK(upload(&query_rope_device, query_rope, stream));
    CHECK(upload(&kv_b_device, kv_b, stream));
    CHECK(upload(&latent_device, latent_first, stream));
    CHECK(upload(&rope_device, rope_first, stream));
    CHECK(cache.append(latent_device, rope_device, stream).is_ok());

    CudaMlaAttentionRequest request{
        static_cast<const float*>(query_nope_device.data()),
        static_cast<const float*>(query_rope_device.data()),
        static_cast<const float*>(kv_b_device.data()),
        static_cast<float*>(probabilities_device.data()),
        static_cast<float*>(output_device.data()),
        heads, nope, rope, value, latent, 2, scale};
    const auto& backend = cpp_llm_engine::cuda::cuda_mla_attention_backend();
    CHECK(backend.compute(request, cache, stream).is_ok());
    std::vector<float> output(heads * value);
    CHECK(output_device.copy_to_host_async(
        output.data(), output.size() * sizeof(float), stream).is_ok());
    CHECK(stream.synchronize().is_ok());
    auto golden = cpu_mla(
        query_nope, query_rope, kv_b, latent_first, rope_first, 1,
        heads, nope, rope, value, latent, scale);
    for (std::size_t index = 0; index < output.size(); ++index) {
        CHECK(std::fabs(output[index] - golden[index]) <= 1.0e-5F);
    }

    CHECK(upload(&latent_device, latent_second, stream));
    CHECK(upload(&rope_device, rope_second, stream));
    CHECK(cache.append(latent_device, rope_device, stream).is_ok());
    CHECK(backend.compute(request, cache, stream).is_ok());
    CHECK(output_device.copy_to_host_async(
        output.data(), output.size() * sizeof(float), stream).is_ok());
    CHECK(stream.synchronize().is_ok());
    std::vector<float> latent_history = latent_first;
    latent_history.insert(
        latent_history.end(), latent_second.begin(), latent_second.end());
    std::vector<float> rope_history = rope_first;
    rope_history.insert(
        rope_history.end(), rope_second.begin(), rope_second.end());
    golden = cpu_mla(
        query_nope, query_rope, kv_b, latent_history, rope_history, 2,
        heads, nope, rope, value, latent, scale);
    for (std::size_t index = 0; index < output.size(); ++index) {
        CHECK(std::fabs(output[index] - golden[index]) <= 1.0e-5F);
    }
    CHECK(cache.size_tokens() == 2);
    CHECK(isolated.size_tokens() == 0);
    CHECK(!cache.append(latent_device, rope_device, stream).is_ok());
    auto overflow_request = request;
    overflow_request.head_count = std::numeric_limits<std::size_t>::max();
    CHECK(!backend.compute(overflow_request, cache, stream).is_ok());
    CHECK(cache.trim(1).is_ok());
    CHECK(cache.size_tokens() == 1);
    cache.clear();
    CHECK(cache.size_tokens() == 0);
    CHECK(!backend.compute(request, cache, stream).is_ok());
    return true;
}

}  // namespace

int main() {
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count <= 0) {
        return 1;
    }
    return run_mla_case() ? 0 : 1;
}

#undef CHECK
