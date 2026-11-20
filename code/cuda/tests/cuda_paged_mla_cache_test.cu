// CUDA 分页 MLA：验证非连续 BlockTable 到压缩设备载荷的真实映射。
// 职责：覆盖双层写入、读回、物理块清零和复用前隔离。
// 通过条件：latent K/V 与 RoPE key 经 block/slot 定位后逐元素一致。
#include "chapter07_kv_cache/block_table.h"
#include "cpp_llm_engine/core/tensor_contract.h"
#include "cpp_llm_engine/cuda/cuda_execution_context.h"
#include "cpp_llm_engine/cuda/cuda_mla_cache.h"
#include "cpp_llm_engine/cuda/cuda_paged_mla_cache.h"
#include "cpp_llm_engine/cuda/cuda_stream.h"
#include "cpp_llm_engine/cuda/cuda_tensor.h"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

#define CHECK(c) do { if (!(c)) { std::cerr << "check failed line " << __LINE__ << '\n'; return 1; } } while (false)

int main() {
    int devices = 0;
    CHECK(cudaGetDeviceCount(&devices) == cudaSuccess && devices > 0);
    constexpr std::size_t layers = 2;
    constexpr std::size_t total_blocks = 3;
    constexpr std::size_t block_size = 2;
    constexpr std::size_t latent_dimension = 4;
    constexpr std::size_t rope_dimension = 2;

    chapter07_kv_cache::KvBlockPool pool(total_blocks, block_size);
    CHECK(pool.reserve(10, 2));
    CHECK(pool.commit_tokens(10, 2));
    CHECK(pool.reserve(20, 2));
    CHECK(pool.commit_tokens(20, 2));
    CHECK(pool.release(10));
    CHECK(pool.reserve(20, 3));
    const auto blocks = pool.physical_blocks(20);
    CHECK(blocks.size() == 2);
    CHECK(blocks[0] == 1 && blocks[1] == 0);

    cpp_llm_engine::cuda::CudaStream stream;
    cpp_llm_engine::cuda::CudaPagedMlaCache cache;
    CHECK(cpp_llm_engine::cuda::CudaStream::create(&stream).is_ok());
    CHECK(cpp_llm_engine::cuda::CudaPagedMlaCache::create(
        layers,
        total_blocks,
        block_size,
        latent_dimension,
        rope_dimension,
        &cache).is_ok());
    CHECK(cache.valid());
    CHECK(cache.payload_bytes() ==
        layers * total_blocks * block_size *
            (latent_dimension + rope_dimension) * sizeof(float));

    cpp_llm_engine::cuda::CudaTensor latent;
    cpp_llm_engine::cuda::CudaTensor key_rope;
    CHECK(cpp_llm_engine::cuda::CudaTensor::create_contiguous(
        {1, latent_dimension},
        cpp_llm_engine::core::DType::Float32,
        &latent).is_ok());
    CHECK(cpp_llm_engine::cuda::CudaTensor::create_contiguous(
        {1, rope_dimension},
        cpp_llm_engine::core::DType::Float32,
        &key_rope).is_ok());

    std::vector<std::vector<float>> expected_latent(layers * 3);
    std::vector<std::vector<float>> expected_rope(layers * 3);
    for (std::size_t token = 0; token < 3; ++token) {
        const auto location = pool.locate_reserved(20, token);
        for (std::size_t layer = 0; layer < layers; ++layer) {
            auto& latent_row = expected_latent[layer * 3 + token];
            auto& rope_row = expected_rope[layer * 3 + token];
            latent_row.resize(latent_dimension);
            rope_row.resize(rope_dimension);
            for (std::size_t index = 0; index < latent_dimension; ++index) {
                latent_row[index] =
                    static_cast<float>(100 * layer + 10 * token + index);
            }
            for (std::size_t index = 0; index < rope_dimension; ++index) {
                rope_row[index] =
                    -static_cast<float>(100 * layer + 10 * token + index + 1);
            }
            CHECK(latent.copy_from_host_async(
                latent_row.data(),
                latent_row.size() * sizeof(float),
                stream).is_ok());
            CHECK(key_rope.copy_from_host_async(
                rope_row.data(),
                rope_row.size() * sizeof(float),
                stream).is_ok());
            CHECK(cache.write_token(
                layer,
                location.block,
                location.slot,
                latent,
                key_rope,
                stream).is_ok());
        }
    }

    constexpr std::size_t head_count = 2;
    constexpr std::size_t nope_dimension = 2;
    constexpr std::size_t value_dimension = 2;
    cpp_llm_engine::cuda::CudaExecutionContext execution;
    cpp_llm_engine::cuda::CudaMlaCache continuous;
    CHECK(cpp_llm_engine::cuda::CudaExecutionContext::create(&execution).is_ok());
    CHECK(cpp_llm_engine::cuda::CudaMlaCache::create(
        3, latent_dimension, rope_dimension, &continuous).is_ok());
    for (std::size_t token = 0; token < 3; ++token) {
        CHECK(execution.upload(
            &latent,
            expected_latent[token].data(),
            latent_dimension * sizeof(float)).is_ok());
        CHECK(execution.upload(
            &key_rope,
            expected_rope[token].data(),
            rope_dimension * sizeof(float)).is_ok());
        CHECK(execution.append_mla(&continuous, latent, key_rope).is_ok());
    }

    cpp_llm_engine::cuda::CudaTensor query_nope;
    cpp_llm_engine::cuda::CudaTensor query_rope;
    cpp_llm_engine::cuda::CudaTensor kv_b;
    cpp_llm_engine::cuda::CudaTensor continuous_probabilities;
    cpp_llm_engine::cuda::CudaTensor paged_probabilities;
    cpp_llm_engine::cuda::CudaTensor continuous_output;
    cpp_llm_engine::cuda::CudaTensor paged_output;
    CHECK(cpp_llm_engine::cuda::CudaTensor::create_contiguous(
        {head_count, nope_dimension},
        cpp_llm_engine::core::DType::Float32,
        &query_nope).is_ok());
    CHECK(cpp_llm_engine::cuda::CudaTensor::create_contiguous(
        {1, head_count, rope_dimension},
        cpp_llm_engine::core::DType::Float32,
        &query_rope).is_ok());
    CHECK(cpp_llm_engine::cuda::CudaTensor::create_contiguous(
        {head_count * (nope_dimension + value_dimension), latent_dimension},
        cpp_llm_engine::core::DType::Float32,
        &kv_b).is_ok());
    CHECK(cpp_llm_engine::cuda::CudaTensor::create_contiguous(
        {head_count, 3},
        cpp_llm_engine::core::DType::Float32,
        &continuous_probabilities).is_ok());
    CHECK(cpp_llm_engine::cuda::CudaTensor::create_contiguous(
        {head_count, 3},
        cpp_llm_engine::core::DType::Float32,
        &paged_probabilities).is_ok());
    CHECK(cpp_llm_engine::cuda::CudaTensor::create_contiguous(
        {head_count, value_dimension},
        cpp_llm_engine::core::DType::Float32,
        &continuous_output).is_ok());
    CHECK(cpp_llm_engine::cuda::CudaTensor::create_contiguous(
        {head_count, value_dimension},
        cpp_llm_engine::core::DType::Float32,
        &paged_output).is_ok());
    const std::vector<float> query_nope_values{0.1F, 0.2F, -0.3F, 0.4F};
    const std::vector<float> query_rope_values{0.5F, -0.2F, 0.3F, 0.1F};
    std::vector<float> kv_b_values(
        head_count * (nope_dimension + value_dimension) * latent_dimension);
    for (std::size_t index = 0; index < kv_b_values.size(); ++index) {
        kv_b_values[index] = static_cast<float>(index + 1) * 0.01F;
    }
    CHECK(execution.upload(
        &query_nope,
        query_nope_values.data(),
        query_nope_values.size() * sizeof(float)).is_ok());
    CHECK(execution.upload(
        &query_rope,
        query_rope_values.data(),
        query_rope_values.size() * sizeof(float)).is_ok());
    CHECK(execution.upload(
        &kv_b,
        kv_b_values.data(),
        kv_b_values.size() * sizeof(float)).is_ok());
    constexpr float scale = 0.5F;
    CHECK(execution.mla_attention(
        query_nope,
        query_rope,
        kv_b,
        continuous,
        &continuous_probabilities,
        &continuous_output,
        scale).is_ok());
    CHECK(execution.paged_mla_attention(
        query_nope,
        query_rope,
        kv_b,
        cache,
        0,
        blocks,
        3,
        &paged_probabilities,
        &paged_output,
        scale).is_ok());
    std::vector<float> continuous_values(head_count * value_dimension);
    std::vector<float> paged_values(head_count * value_dimension);
    CHECK(execution.download(
        continuous_output,
        continuous_values.data(),
        continuous_values.size() * sizeof(float)).is_ok());
    CHECK(execution.download(
        paged_output,
        paged_values.data(),
        paged_values.size() * sizeof(float)).is_ok());
    CHECK(execution.synchronize().is_ok());
    for (std::size_t index = 0; index < continuous_values.size(); ++index) {
        CHECK(std::fabs(continuous_values[index] - paged_values[index]) <= 1.0e-5F);
    }

    cpp_llm_engine::cuda::CudaTensor batch_query_nope;
    cpp_llm_engine::cuda::CudaTensor batch_query_rope;
    cpp_llm_engine::cuda::CudaTensor continuous_batch_probabilities;
    cpp_llm_engine::cuda::CudaTensor paged_batch_probabilities;
    cpp_llm_engine::cuda::CudaTensor continuous_batch_output;
    cpp_llm_engine::cuda::CudaTensor paged_batch_output;
    CHECK(cpp_llm_engine::cuda::CudaTensor::create_contiguous(
        {2, head_count, nope_dimension},
        cpp_llm_engine::core::DType::Float32,
        &batch_query_nope).is_ok());
    CHECK(cpp_llm_engine::cuda::CudaTensor::create_contiguous(
        {2, head_count, rope_dimension},
        cpp_llm_engine::core::DType::Float32,
        &batch_query_rope).is_ok());
    CHECK(cpp_llm_engine::cuda::CudaTensor::create_contiguous(
        {2, head_count, 3}, cpp_llm_engine::core::DType::Float32,
        &continuous_batch_probabilities).is_ok());
    CHECK(cpp_llm_engine::cuda::CudaTensor::create_contiguous(
        {2, head_count, 3}, cpp_llm_engine::core::DType::Float32,
        &paged_batch_probabilities).is_ok());
    CHECK(cpp_llm_engine::cuda::CudaTensor::create_contiguous(
        {2, head_count, value_dimension},
        cpp_llm_engine::core::DType::Float32,
        &continuous_batch_output).is_ok());
    CHECK(cpp_llm_engine::cuda::CudaTensor::create_contiguous(
        {2, head_count, value_dimension},
        cpp_llm_engine::core::DType::Float32,
        &paged_batch_output).is_ok());
    std::vector<float> batch_query_nope_values = query_nope_values;
    batch_query_nope_values.insert(batch_query_nope_values.end(),
        query_nope_values.begin(), query_nope_values.end());
    std::vector<float> batch_query_rope_values = query_rope_values;
    batch_query_rope_values.insert(batch_query_rope_values.end(),
        query_rope_values.begin(), query_rope_values.end());
    CHECK(execution.upload(&batch_query_nope, batch_query_nope_values.data(),
        batch_query_nope_values.size() * sizeof(float)).is_ok());
    CHECK(execution.upload(&batch_query_rope, batch_query_rope_values.data(),
        batch_query_rope_values.size() * sizeof(float)).is_ok());
    CHECK(execution.mla_attention_batch(
        batch_query_nope, batch_query_rope, kv_b, continuous, 1,
        &continuous_batch_probabilities, &continuous_batch_output,
        scale).is_ok());
    CHECK(execution.paged_mla_attention_batch(
        batch_query_nope, batch_query_rope, kv_b, cache, 0, blocks, 3, 1,
        &paged_batch_probabilities, &paged_batch_output, scale).is_ok());
    std::vector<float> continuous_batch_values(
        2 * head_count * value_dimension);
    std::vector<float> paged_batch_values(
        2 * head_count * value_dimension);
    CHECK(execution.download(continuous_batch_output,
        continuous_batch_values.data(),
        continuous_batch_values.size() * sizeof(float)).is_ok());
    CHECK(execution.download(paged_batch_output, paged_batch_values.data(),
        paged_batch_values.size() * sizeof(float)).is_ok());
    CHECK(execution.synchronize().is_ok());
    for (std::size_t index = 0; index < continuous_batch_values.size();
         ++index) {
        CHECK(std::fabs(continuous_batch_values[index] -
            paged_batch_values[index]) <= 1.0e-5F);
    }

    cpp_llm_engine::cuda::CudaTensor latent_batch;
    cpp_llm_engine::cuda::CudaTensor rope_batch;
    CHECK(cpp_llm_engine::cuda::CudaTensor::create_contiguous(
        {3, latent_dimension}, cpp_llm_engine::core::DType::Float32,
        &latent_batch).is_ok());
    CHECK(cpp_llm_engine::cuda::CudaTensor::create_contiguous(
        {3, rope_dimension}, cpp_llm_engine::core::DType::Float32,
        &rope_batch).is_ok());
    std::vector<float> layer_one_latent;
    std::vector<float> layer_one_rope;
    for (std::size_t token = 0; token < 3; ++token) {
        const auto& latent_row = expected_latent[3 + token];
        const auto& rope_row = expected_rope[3 + token];
        layer_one_latent.insert(
            layer_one_latent.end(), latent_row.begin(), latent_row.end());
        layer_one_rope.insert(
            layer_one_rope.end(), rope_row.begin(), rope_row.end());
    }
    CHECK(execution.upload(&latent_batch, layer_one_latent.data(),
        layer_one_latent.size() * sizeof(float)).is_ok());
    CHECK(execution.upload(&rope_batch, layer_one_rope.data(),
        layer_one_rope.size() * sizeof(float)).is_ok());
    CHECK(execution.write_paged_mla_batch(
        &cache, 1, blocks, 0, latent_batch, rope_batch).is_ok());
    CHECK(!cache.write_token(
        layers, 0, 0, latent, key_rope, stream).is_ok());
    CHECK(!cache.write_token(
        0, total_blocks, 0, latent, key_rope, stream).is_ok());

    for (std::size_t token = 0; token < 3; ++token) {
        const auto location = pool.locate_reserved(20, token);
        for (std::size_t layer = 0; layer < layers; ++layer) {
            std::vector<float> actual_latent(latent_dimension);
            std::vector<float> actual_rope(rope_dimension);
            CHECK(cache.copy_token_to_host_async(
                layer,
                location.block,
                location.slot,
                actual_latent.data(),
                actual_rope.data(),
                stream).is_ok());
            CHECK(stream.synchronize().is_ok());
            CHECK(actual_latent == expected_latent[layer * 3 + token]);
            CHECK(actual_rope == expected_rope[layer * 3 + token]);
        }
    }

    CHECK(cache.clear_block(blocks[1], stream).is_ok());
    CHECK(stream.synchronize().is_ok());
    for (std::size_t layer = 0; layer < layers; ++layer) {
        for (std::size_t slot = 0; slot < block_size; ++slot) {
            std::vector<float> actual_latent(latent_dimension, 1.0F);
            std::vector<float> actual_rope(rope_dimension, 1.0F);
            CHECK(cache.copy_token_to_host_async(
                layer,
                blocks[1],
                slot,
                actual_latent.data(),
                actual_rope.data(),
                stream).is_ok());
            CHECK(stream.synchronize().is_ok());
            CHECK(std::all_of(
                actual_latent.begin(), actual_latent.end(),
                [](float value) { return value == 0.0F; }));
            CHECK(std::all_of(
                actual_rope.begin(), actual_rope.end(),
                [](float value) { return value == 0.0F; }));
        }
    }

    CHECK(cache.reset().is_ok());
    CHECK(!cache.valid());
    std::cout << "cuda_paged_mla_noncontiguous_blocks: true\n"
              << "cuda_paged_mla_matches_continuous: true\n"
              << "cuda_paged_mla_batch_matches_continuous: true\n"
              << "cuda_paged_mla_clear_before_reuse: true\n";
    return 0;
}

#undef CHECK
