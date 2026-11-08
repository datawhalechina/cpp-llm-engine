// CUDA Backend：验证三阶段与融合在线 GQA 的 prefill、decode、head 映射和计时。
// 职责：用独立 CPU 公式检查连续 KV、稳定 softmax 和在线状态合并。
// 通过条件：两条 CUDA 路径数值通过，错误 shape 被拒绝并输出 A/B event 均值。
#include "cpp_llm_engine/core/gqa_attention_contract.h"
#include "cpp_llm_engine/core/tensor_contract.h"
#include "cpp_llm_engine/cuda/cuda_event.h"
#include "cpp_llm_engine/cuda/cuda_execution_context.h"
#include "cpp_llm_engine/cuda/cuda_kv_cache.h"
#include "cpp_llm_engine/cuda/cuda_stream.h"
#include "cpp_llm_engine/cuda/cuda_tensor.h"
#include "cpp_llm_engine/cuda/device_buffer.h"
#include "cpp_llm_engine/cuda/gqa_attention_backend.h"

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
            std::cerr << "check failed: " #condition " at line " << __LINE__ \
                      << '\n';                                                  \
            return false;                                                       \
        }                                                                       \
    } while (false)

namespace {

using cpp_llm_engine::core::DType;
using cpp_llm_engine::core::GqaAttentionRequest;
using cpp_llm_engine::cuda::CudaEvent;
using cpp_llm_engine::cuda::CudaExecutionContext;
using cpp_llm_engine::cuda::CudaKvCache;
using cpp_llm_engine::cuda::CudaStream;
using cpp_llm_engine::cuda::CudaTensor;
using cpp_llm_engine::cuda::DeviceBuffer;

std::vector<float> cpu_gqa(
    const std::vector<float>& query,
    const std::vector<float>& key,
    const std::vector<float>& value,
    std::size_t query_tokens,
    std::size_t key_tokens,
    std::size_t query_heads,
    std::size_t kv_heads,
    std::size_t head_dimension,
    std::size_t position_offset) {
    std::vector<float> output(query.size(), 0.0F);
    std::vector<float> scores(key_tokens);
    const std::size_t group_size = query_heads / kv_heads;
    const float scale = 1.0F / std::sqrt(static_cast<float>(head_dimension));
    for (std::size_t token = 0; token < query_tokens; ++token) {
        for (std::size_t head = 0; head < query_heads; ++head) {
            const std::size_t kv_head = head / group_size;
            const float* query_vector = query.data() +
                (token * query_heads + head) * head_dimension;
            for (std::size_t key_token = 0; key_token < key_tokens; ++key_token) {
                if (key_token > position_offset + token) {
                    scores[key_token] = -std::numeric_limits<float>::infinity();
                    continue;
                }
                const float* key_vector = key.data() +
                    (key_token * kv_heads + kv_head) * head_dimension;
                float score = 0.0F;
                for (std::size_t dimension = 0; dimension < head_dimension; ++dimension) {
                    score += query_vector[dimension] * key_vector[dimension];
                }
                scores[key_token] = score * scale;
            }
            const float maximum = *std::max_element(scores.begin(), scores.end());
            float denominator = 0.0F;
            for (float& score : scores) {
                score = std::isinf(score) && score < 0.0F
                            ? 0.0F
                            : std::exp(score - maximum);
                denominator += score;
            }
            for (std::size_t dimension = 0; dimension < head_dimension; ++dimension) {
                float result = 0.0F;
                for (std::size_t key_token = 0; key_token < key_tokens; ++key_token) {
                    const std::size_t value_index =
                        (key_token * kv_heads + kv_head) * head_dimension + dimension;
                    result += scores[key_token] / denominator * value[value_index];
                }
                output[(token * query_heads + head) * head_dimension + dimension] = result;
            }
        }
    }
    return output;
}

bool create_and_upload(
    CudaExecutionContext* context,
    const std::vector<float>& values,
    const std::vector<std::size_t>& shape,
    CudaTensor* output) {
    CHECK(CudaTensor::create_contiguous(shape, DType::Float32, output).is_ok());
    CHECK(context->upload(output, values.data(), values.size() * sizeof(float)).is_ok());
    return true;
}

bool compare_output(
    CudaExecutionContext* context,
    const CudaTensor& tensor,
    const std::vector<float>& golden) {
    std::vector<float> output(golden.size());
    CHECK(context->download(
        tensor, output.data(), output.size() * sizeof(float)).is_ok());
    CHECK(context->synchronize().is_ok());
    for (std::size_t index = 0; index < output.size(); ++index) {
        CHECK(std::fabs(output[index] - golden[index]) <= 3.0e-5F);
    }
    return true;
}

bool test_prefill_and_decode() {
    constexpr std::size_t query_heads = 4;
    constexpr std::size_t kv_heads = 2;
    constexpr std::size_t head_dimension = 3;
    CudaExecutionContext context;
    CudaKvCache cache;
    CHECK(CudaExecutionContext::create(&context).is_ok());
    CHECK(CudaKvCache::create(5, kv_heads, head_dimension, &cache).is_ok());

    std::vector<float> query(3 * query_heads * head_dimension);
    std::vector<float> key(3 * kv_heads * head_dimension);
    std::vector<float> value(key.size());
    for (std::size_t index = 0; index < query.size(); ++index) {
        query[index] = static_cast<float>(static_cast<int>(index % 11) - 5) * 0.1F;
    }
    for (std::size_t index = 0; index < key.size(); ++index) {
        key[index] = static_cast<float>(static_cast<int>(index % 7) - 3) * 0.2F;
        value[index] = static_cast<float>(index + 1) * 0.05F;
    }
    CudaTensor query_tensor;
    CudaTensor key_tensor;
    CudaTensor value_tensor;
    CudaTensor output;
    CHECK(create_and_upload(
        &context, query, {3, query_heads, head_dimension}, &query_tensor));
    CHECK(create_and_upload(
        &context, key, {3, kv_heads, head_dimension}, &key_tensor));
    CHECK(create_and_upload(
        &context, value, {3, kv_heads, head_dimension}, &value_tensor));
    CHECK(CudaTensor::create_contiguous(
        {3, query_heads, head_dimension}, DType::Float32, &output).is_ok());
    CHECK(context.append_kv(&cache, key_tensor, value_tensor).is_ok());
    CHECK(context.gqa_attention(query_tensor, cache, &output, 0).is_ok());
    CHECK(compare_output(
        &context, output,
        cpu_gqa(query, key, value, 3, 3, query_heads, kv_heads, head_dimension, 0)));

    std::vector<float> decode_query(query_heads * head_dimension);
    std::vector<float> decode_key(kv_heads * head_dimension);
    std::vector<float> decode_value(kv_heads * head_dimension);
    for (std::size_t index = 0; index < decode_query.size(); ++index) {
        decode_query[index] = static_cast<float>(index + 2) * -0.07F;
    }
    for (std::size_t index = 0; index < decode_key.size(); ++index) {
        decode_key[index] = static_cast<float>(index + 1) * 0.11F;
        decode_value[index] = static_cast<float>(index + 3) * -0.09F;
    }
    CudaTensor decode_query_tensor;
    CudaTensor decode_key_tensor;
    CudaTensor decode_value_tensor;
    CudaTensor decode_output;
    CHECK(create_and_upload(
        &context, decode_query, {1, query_heads, head_dimension},
        &decode_query_tensor));
    CHECK(create_and_upload(
        &context, decode_key, {1, kv_heads, head_dimension}, &decode_key_tensor));
    CHECK(create_and_upload(
        &context, decode_value, {1, kv_heads, head_dimension},
        &decode_value_tensor));
    CHECK(CudaTensor::create_contiguous(
        {1, query_heads, head_dimension}, DType::Float32,
        &decode_output).is_ok());
    CHECK(context.append_kv(&cache, decode_key_tensor, decode_value_tensor).is_ok());
    CHECK(context.gqa_attention(
        decode_query_tensor, cache, &decode_output, 3).is_ok());
    key.insert(key.end(), decode_key.begin(), decode_key.end());
    value.insert(value.end(), decode_value.begin(), decode_value.end());
    CHECK(compare_output(
        &context, decode_output,
        cpu_gqa(
            decode_query, key, value, 1, 4, query_heads, kv_heads,
            head_dimension, 3)));
    return true;
}

bool test_errors() {
    CudaExecutionContext context;
    CudaKvCache cache;
    CHECK(CudaExecutionContext::create(&context).is_ok());
    CHECK(CudaKvCache::create(2, 2, 4, &cache).is_ok());
    CudaTensor query;
    CudaTensor output;
    CudaTensor key;
    const std::vector<float> query_values(3 * 4, 1.0F);
    const std::vector<float> key_values(2 * 4, 1.0F);
    CHECK(create_and_upload(&context, query_values, {1, 3, 4}, &query));
    CHECK(create_and_upload(&context, key_values, {1, 2, 4}, &key));
    CHECK(CudaTensor::create_contiguous({1, 3, 4}, DType::Float32, &output).is_ok());
    CHECK(!context.gqa_attention(query, cache, &output, 0).is_ok());
    CHECK(context.append_kv(&cache, key, key).is_ok());
    CHECK(!context.gqa_attention(query, cache, &output, 0).is_ok());
    CHECK(!context.gqa_attention(query, cache, nullptr, 0).is_ok());
    CHECK(!context.append_kv(nullptr, key, key).is_ok());

    const auto& backend = cpp_llm_engine::cuda::cuda_gqa_attention_backend();
    CudaStream invalid_stream;
    float fake = 0.0F;
    const GqaAttentionRequest request{
        &fake, &fake, &fake, &fake, 1, 1, 2, 1, 1, 0, true, 0.0F};
    CHECK(backend.kind() == cpp_llm_engine::core::BackendKind::CUDA);
    CHECK(backend.name() == "cuda-gqa-attention");
    CHECK(!backend.compute(request, &fake, 1, invalid_stream).is_ok());
    return true;
}

bool test_fused_online_prefill_and_decode() {
    const auto run_case = [](
                              std::size_t query_tokens,
                              std::size_t key_tokens,
                              std::size_t position_offset) {
        constexpr std::size_t query_heads = 4;
        constexpr std::size_t kv_heads = 2;
        constexpr std::size_t head_dimension = 64;
        const std::size_t query_elements = query_tokens * query_heads * head_dimension;
        const std::size_t kv_elements = key_tokens * kv_heads * head_dimension;
        std::vector<float> query_host(query_elements);
        std::vector<float> key_host(kv_elements);
        std::vector<float> value_host(kv_elements);
        for (std::size_t index = 0; index < query_elements; ++index) {
            query_host[index] =
                static_cast<float>(static_cast<int>(index % 17) - 8) * 0.03125F;
        }
        for (std::size_t index = 0; index < kv_elements; ++index) {
            key_host[index] =
                static_cast<float>(static_cast<int>(index % 13) - 6) * 0.025F;
            value_host[index] =
                static_cast<float>(static_cast<int>(index % 19) - 9) * 0.05F;
        }
        const auto golden = cpu_gqa(
            query_host, key_host, value_host, query_tokens, key_tokens,
            query_heads, kv_heads, head_dimension, position_offset);

        DeviceBuffer query;
        DeviceBuffer key;
        DeviceBuffer value;
        DeviceBuffer output;
        CudaStream stream;
        CHECK(DeviceBuffer::allocate(query_elements * sizeof(float), &query).is_ok());
        CHECK(DeviceBuffer::allocate(kv_elements * sizeof(float), &key).is_ok());
        CHECK(DeviceBuffer::allocate(kv_elements * sizeof(float), &value).is_ok());
        CHECK(DeviceBuffer::allocate(query_elements * sizeof(float), &output).is_ok());
        CHECK(CudaStream::create(&stream).is_ok());
        CHECK(query.copy_from_host_async(
            query_host.data(), query_elements * sizeof(float), stream).is_ok());
        CHECK(key.copy_from_host_async(
            key_host.data(), kv_elements * sizeof(float), stream).is_ok());
        CHECK(value.copy_from_host_async(
            value_host.data(), kv_elements * sizeof(float), stream).is_ok());
        const GqaAttentionRequest request{
            static_cast<const float*>(query.data()),
            static_cast<const float*>(key.data()),
            static_cast<const float*>(value.data()),
            static_cast<float*>(output.data()),
            query_tokens, key_tokens, query_heads, kv_heads, head_dimension,
            position_offset, true, 0.0F};
        CHECK(cpp_llm_engine::cuda::cuda_gqa_attention_backend()
                  .compute_online(request, stream)
                  .is_ok());
        std::vector<float> actual(query_elements);
        CHECK(output.copy_to_host_async(
            actual.data(), query_elements * sizeof(float), stream).is_ok());
        CHECK(stream.synchronize().is_ok());
        for (std::size_t index = 0; index < actual.size(); ++index) {
            CHECK(std::fabs(actual[index] - golden[index]) <= 4.0e-5F);
        }

        auto unsupported = request;
        unsupported.head_dimension = 129;
        CHECK(!cpp_llm_engine::cuda::cuda_gqa_attention_backend()
                   .compute_online(unsupported, stream)
                   .is_ok());
        return true;
    };

    CHECK(run_case(5, 7, 2));
    CHECK(run_case(1, 7, 6));
    return true;
}

bool run_benchmark() {
    constexpr std::size_t query_tokens = 17;
    constexpr std::size_t key_tokens = 257;
    constexpr std::size_t query_heads = 32;
    constexpr std::size_t kv_heads = 4;
    constexpr std::size_t head_dimension = 64;
    constexpr int warmup = 1;
    constexpr int iterations = 10;
    const std::size_t query_elements = query_tokens * query_heads * head_dimension;
    const std::size_t kv_elements = key_tokens * kv_heads * head_dimension;
    const std::size_t score_elements = query_tokens * query_heads * key_tokens;
    std::vector<float> query_host(query_elements, 0.01F);
    std::vector<float> kv_host(kv_elements, 0.02F);
    DeviceBuffer query;
    DeviceBuffer key;
    DeviceBuffer value;
    DeviceBuffer output;
    DeviceBuffer workspace;
    CudaStream stream;
    CudaEvent start;
    CudaEvent end;
    CHECK(DeviceBuffer::allocate(query_elements * sizeof(float), &query).is_ok());
    CHECK(DeviceBuffer::allocate(kv_elements * sizeof(float), &key).is_ok());
    CHECK(DeviceBuffer::allocate(kv_elements * sizeof(float), &value).is_ok());
    CHECK(DeviceBuffer::allocate(query_elements * sizeof(float), &output).is_ok());
    CHECK(DeviceBuffer::allocate(score_elements * sizeof(float), &workspace).is_ok());
    CHECK(CudaStream::create(&stream).is_ok());
    CHECK(CudaEvent::create(&start).is_ok());
    CHECK(CudaEvent::create(&end).is_ok());
    CHECK(query.copy_from_host_async(
        query_host.data(), query_elements * sizeof(float), stream).is_ok());
    CHECK(key.copy_from_host_async(
        kv_host.data(), kv_elements * sizeof(float), stream).is_ok());
    CHECK(value.copy_from_host_async(
        kv_host.data(), kv_elements * sizeof(float), stream).is_ok());
    const GqaAttentionRequest request{
        static_cast<const float*>(query.data()),
        static_cast<const float*>(key.data()),
        static_cast<const float*>(value.data()),
        static_cast<float*>(output.data()),
        query_tokens, key_tokens, query_heads, kv_heads, head_dimension,
        240, true, 0.0F};
    for (int iteration = 0; iteration < warmup; ++iteration) {
        CHECK(cpp_llm_engine::cuda::cuda_gqa_attention_backend().compute(
            request, static_cast<float*>(workspace.data()), score_elements,
            stream).is_ok());
    }
    CHECK(stream.synchronize().is_ok());
    CHECK(start.record(stream).is_ok());
    for (int iteration = 0; iteration < iterations; ++iteration) {
        CHECK(cpp_llm_engine::cuda::cuda_gqa_attention_backend().compute(
            request, static_cast<float*>(workspace.data()), score_elements,
            stream).is_ok());
    }
    CHECK(end.record(stream).is_ok());
    CHECK(end.synchronize().is_ok());
    float three_stage_elapsed_ms = 0.0F;
    CHECK(CudaEvent::elapsed_time_ms(start, end, &three_stage_elapsed_ms).is_ok());

    for (int iteration = 0; iteration < warmup; ++iteration) {
        CHECK(cpp_llm_engine::cuda::cuda_gqa_attention_backend()
                  .compute_online(request, stream)
                  .is_ok());
    }
    CHECK(stream.synchronize().is_ok());
    CHECK(start.record(stream).is_ok());
    for (int iteration = 0; iteration < iterations; ++iteration) {
        CHECK(cpp_llm_engine::cuda::cuda_gqa_attention_backend()
                  .compute_online(request, stream)
                  .is_ok());
    }
    CHECK(end.record(stream).is_ok());
    CHECK(end.synchronize().is_ok());
    float fused_elapsed_ms = 0.0F;
    CHECK(CudaEvent::elapsed_time_ms(start, end, &fused_elapsed_ms).is_ok());
    std::cout << "gqa_query_tokens=" << query_tokens
              << " key_tokens=" << key_tokens
              << " query_heads=" << query_heads
              << " kv_heads=" << kv_heads
              << " head_dimension=" << head_dimension
              << " warmup=" << warmup
              << " iterations=" << iterations
              << " three_stage_mean_ms=" << three_stage_elapsed_ms / iterations
              << " fused_online_mean_ms=" << fused_elapsed_ms / iterations << '\n';
    return true;
}

}  // namespace

int main() {
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count <= 0) return 1;
    if (!test_errors()) return 1;
    if (!test_prefill_and_decode()) return 1;
    if (!test_fused_online_prefill_and_decode()) return 1;
    if (!run_benchmark()) return 1;
    return 0;
}

#undef CHECK
