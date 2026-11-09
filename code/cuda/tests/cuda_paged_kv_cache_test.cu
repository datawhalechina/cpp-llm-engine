// CUDA 分页 K/V：验证 Chapter07 非连续 BlockTable 到设备 payload 的真实映射。
// 职责：覆盖双层写入/读回、三阶段/融合 Attention、物理块清零和复用。
// 通过条件：逻辑 token 经 block/slot 定位后与 CPU golden 逐元素一致。
#include "chapter07_kv_cache/block_table.h"
#include "cpp_llm_engine/core/tensor_contract.h"
#include "cpp_llm_engine/cuda/cuda_paged_kv_cache.h"
#include "cpp_llm_engine/cuda/cuda_execution_context.h"
#include "cpp_llm_engine/cuda/cuda_stream.h"
#include "cpp_llm_engine/cuda/cuda_tensor.h"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

#define CHECK(c) do { if (!(c)) { std::cerr << "check failed line " << __LINE__ << '\n'; return 1; } } while (false)

namespace {

std::vector<float> cpu_decode_attention(
    const std::vector<float>& query,
    const std::vector<std::vector<float>>& keys,
    const std::vector<std::vector<float>>& values,
    std::size_t query_heads,
    std::size_t kv_heads,
    std::size_t head_dimension) {
    std::vector<float> output(query.size(), 0.0F);
    std::vector<float> scores(keys.size());
    const std::size_t group_size = query_heads / kv_heads;
    const float scale = 1.0F / std::sqrt(static_cast<float>(head_dimension));
    for (std::size_t query_head = 0; query_head < query_heads; ++query_head) {
        const std::size_t kv_head = query_head / group_size;
        for (std::size_t token = 0; token < keys.size(); ++token) {
            float score = 0.0F;
            for (std::size_t dimension = 0; dimension < head_dimension; ++dimension) {
                score += query[query_head * head_dimension + dimension] *
                    keys[token][kv_head * head_dimension + dimension];
            }
            scores[token] = score * scale;
        }
        const float maximum = *std::max_element(scores.begin(), scores.end());
        float denominator = 0.0F;
        for (float& score : scores) {
            score = std::exp(score - maximum);
            denominator += score;
        }
        for (std::size_t dimension = 0; dimension < head_dimension; ++dimension) {
            float result = 0.0F;
            for (std::size_t token = 0; token < values.size(); ++token) {
                result += scores[token] / denominator *
                    values[token][kv_head * head_dimension + dimension];
            }
            output[query_head * head_dimension + dimension] = result;
        }
    }
    return output;
}

}  // namespace

int main() {
    int devices = 0;
    CHECK(cudaGetDeviceCount(&devices) == cudaSuccess && devices > 0);
    constexpr std::size_t layers = 2;
    constexpr std::size_t total_blocks = 4;
    constexpr std::size_t block_size = 2;
    constexpr std::size_t kv_heads = 2;
    constexpr std::size_t head_dimension = 3;
    constexpr std::size_t token_width = kv_heads * head_dimension;

    chapter07_kv_cache::KvBlockPool pool(total_blocks, block_size);
    CHECK(pool.reserve(10, 2));
    CHECK(pool.commit_tokens(10, 2));
    CHECK(pool.reserve(20, 2));
    CHECK(pool.commit_tokens(20, 2));
    CHECK(pool.reserve(30, 2));
    CHECK(pool.commit_tokens(30, 2));
    CHECK(pool.release(10));
    CHECK(pool.reserve(20, 5));
    const auto blocks = pool.physical_blocks(20);
    CHECK(blocks.size() == 3);
    CHECK(blocks[0] == 1 && blocks[1] == 0 && blocks[2] == 3);

    cpp_llm_engine::cuda::CudaStream stream;
    cpp_llm_engine::cuda::CudaPagedKvCache cache;
    CHECK(cpp_llm_engine::cuda::CudaStream::create(&stream).is_ok());
    CHECK(cpp_llm_engine::cuda::CudaPagedKvCache::create(
        layers, total_blocks, block_size, kv_heads, head_dimension, &cache).is_ok());
    CHECK(cache.valid());
    CHECK(cache.payload_bytes() ==
          layers * total_blocks * block_size * token_width * sizeof(float) * 2);

    cpp_llm_engine::cuda::CudaTensor key;
    cpp_llm_engine::cuda::CudaTensor value;
    CHECK(cpp_llm_engine::cuda::CudaTensor::create_contiguous(
        {1, kv_heads, head_dimension}, cpp_llm_engine::core::DType::Float32,
        &key).is_ok());
    CHECK(cpp_llm_engine::cuda::CudaTensor::create_contiguous(
        {1, kv_heads, head_dimension}, cpp_llm_engine::core::DType::Float32,
        &value).is_ok());

    std::vector<std::vector<float>> expected_keys(layers * 5);
    std::vector<std::vector<float>> expected_values(layers * 5);
    for (std::size_t token = 0; token < 5; ++token) {
        const auto location = pool.locate_reserved(20, token);
        for (std::size_t layer = 0; layer < layers; ++layer) {
            auto& key_row = expected_keys[layer * 5 + token];
            auto& value_row = expected_values[layer * 5 + token];
            key_row.resize(token_width);
            value_row.resize(token_width);
            for (std::size_t index = 0; index < token_width; ++index) {
                key_row[index] = static_cast<float>(100 * layer + 10 * token + index);
                value_row[index] = -key_row[index] - 0.5F;
            }
            CHECK(key.copy_from_host_async(
                key_row.data(), key_row.size() * sizeof(float), stream).is_ok());
            CHECK(value.copy_from_host_async(
                value_row.data(), value_row.size() * sizeof(float), stream).is_ok());
            CHECK(cache.write_token(
                layer, location.block, location.slot, key, value, stream).is_ok());
        }
    }
    CHECK(!cache.write_token(layers, 0, 0, key, value, stream).is_ok());
    CHECK(!cache.write_token(0, total_blocks, 0, key, value, stream).is_ok());
    CHECK(!cache.write_token(0, 0, block_size, key, value, stream).is_ok());

    for (std::size_t token = 0; token < 5; ++token) {
        const auto location = pool.locate_reserved(20, token);
        for (std::size_t layer = 0; layer < layers; ++layer) {
            std::vector<float> actual_key(token_width);
            std::vector<float> actual_value(token_width);
            CHECK(cache.copy_token_to_host_async(
                layer, location.block, location.slot,
                actual_key.data(), actual_value.data(), stream).is_ok());
            CHECK(stream.synchronize().is_ok());
            CHECK(actual_key == expected_keys[layer * 5 + token]);
            CHECK(actual_value == expected_values[layer * 5 + token]);
        }
    }

    constexpr std::size_t query_heads = 4;
    std::vector<float> query_values(query_heads * head_dimension);
    for (std::size_t index = 0; index < query_values.size(); ++index) {
        query_values[index] = static_cast<float>(index + 1) * 0.03F;
    }
    cpp_llm_engine::cuda::CudaExecutionContext context;
    cpp_llm_engine::cuda::CudaTensor query;
    cpp_llm_engine::cuda::CudaTensor output;
    CHECK(cpp_llm_engine::cuda::CudaExecutionContext::create(&context).is_ok());
    CHECK(cpp_llm_engine::cuda::CudaTensor::create_contiguous(
        {1, query_heads, head_dimension}, cpp_llm_engine::core::DType::Float32,
        &query).is_ok());
    CHECK(cpp_llm_engine::cuda::CudaTensor::create_contiguous(
        {1, query_heads, head_dimension}, cpp_llm_engine::core::DType::Float32,
        &output).is_ok());
    CHECK(context.upload(
        &query, query_values.data(), query_values.size() * sizeof(float)).is_ok());
    for (std::size_t layer = 0; layer < layers; ++layer) {
        std::vector<std::vector<float>> layer_keys;
        std::vector<std::vector<float>> layer_values;
        for (std::size_t token = 0; token < 5; ++token) {
            layer_keys.push_back(expected_keys[layer * 5 + token]);
            layer_values.push_back(expected_values[layer * 5 + token]);
        }
        const auto golden = cpu_decode_attention(
            query_values, layer_keys, layer_values,
            query_heads, kv_heads, head_dimension);
        CHECK(context.paged_gqa_attention(
            query, cache, layer, blocks, 5, &output, 4).is_ok());
        std::vector<float> actual(golden.size());
        CHECK(context.download(
            output, actual.data(), actual.size() * sizeof(float)).is_ok());
        CHECK(context.synchronize().is_ok());
        for (std::size_t index = 0; index < actual.size(); ++index) {
            CHECK(std::fabs(actual[index] - golden[index]) <= 3.0e-5F);
        }
        context.set_attention_path(
            cpp_llm_engine::cuda::CudaAttentionPath::FusedOnline);
        CHECK(context.paged_gqa_attention(
            query, cache, layer, blocks, 5, &output, 4).is_ok());
        CHECK(context.download(
            output, actual.data(), actual.size() * sizeof(float)).is_ok());
        CHECK(context.synchronize().is_ok());
        for (std::size_t index = 0; index < actual.size(); ++index) {
            CHECK(std::fabs(actual[index] - golden[index]) <= 4.0e-5F);
        }
        context.set_attention_path(
            cpp_llm_engine::cuda::CudaAttentionPath::ThreeStage);
    }
    CHECK(!context.paged_gqa_attention(
        query, cache, 0, {total_blocks}, 1, &output, 0).is_ok());

    CHECK(cache.clear_block(blocks[1], stream).is_ok());
    CHECK(stream.synchronize().is_ok());
    for (std::size_t layer = 0; layer < layers; ++layer) {
        for (std::size_t slot = 0; slot < block_size; ++slot) {
            std::vector<float> actual_key(token_width, 1.0F);
            std::vector<float> actual_value(token_width, 1.0F);
            CHECK(cache.copy_token_to_host_async(
                layer, blocks[1], slot,
                actual_key.data(), actual_value.data(), stream).is_ok());
            CHECK(stream.synchronize().is_ok());
            CHECK(std::all_of(actual_key.begin(), actual_key.end(),
                [](float value) { return value == 0.0F; }));
            CHECK(std::all_of(actual_value.begin(), actual_value.end(),
                [](float value) { return value == 0.0F; }));
        }
    }

    CHECK(cache.reset().is_ok());
    CHECK(!cache.valid());
    std::cout << "cuda_paged_kv_noncontiguous_blocks: true\n"
              << "cuda_paged_gqa_cpu_golden: true\n"
              << "cuda_paged_gqa_fused_online: true\n"
              << "physical_blocks: " << blocks[0] << ' ' << blocks[1] << ' '
              << blocks[2] << '\n';
    return 0;
}

#undef CHECK
