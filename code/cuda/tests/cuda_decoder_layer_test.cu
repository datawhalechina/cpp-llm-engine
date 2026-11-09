// CUDA Decoder：验证单 token、多请求 Decode 与连续/分页 Tensor Prefill。
// 职责：用小维度权重检查算子组装、因果 K/V、页表寻址和事务回滚。
// 通过条件：批量路径与顺序执行一致，分页 chunk 跨块后输出不变。
#include "cpp_llm_engine/core/tensor_contract.h"
#include "cpp_llm_engine/cuda/cuda_decoder_batch_layer.h"
#include "cpp_llm_engine/cuda/cuda_decoder_layer.h"
#include "cpp_llm_engine/cuda/cuda_execution_context.h"
#include "cpp_llm_engine/cuda/cuda_kv_cache.h"
#include "cpp_llm_engine/cuda/cuda_paged_kv_cache.h"

#include <cuda_runtime_api.h>

#include <cmath>
#include <cstddef>
#include <iostream>
#include <vector>

#define CHECK(c) do { if (!(c)) { std::cerr << "check failed line " << __LINE__ << '\n'; return 1; } } while (false)

namespace {
using cpp_llm_engine::core::DType;
using namespace cpp_llm_engine::cuda;

std::vector<float> rms(const std::vector<float>& x) {
    float sum = 0.0F;
    for (float v : x) sum += v * v;
    const float inv = 1.0F / std::sqrt(sum / x.size() + 1.0e-5F);
    std::vector<float> y(x.size());
    for (std::size_t i = 0; i < x.size(); ++i) y[i] = x[i] * inv;
    return y;
}

bool close(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) return false;
    for (std::size_t i = 0; i < a.size(); ++i) {
        if (std::fabs(a[i] - b[i]) > 4.0e-5F) return false;
    }
    return true;
}

}  // namespace

int main() {
    int devices = 0;
    CHECK(cudaGetDeviceCount(&devices) == cudaSuccess && devices > 0);
    CudaExecutionContext context;
    CHECK(CudaExecutionContext::create(&context).is_ok());
    const CudaDecoderLayerConfig config{8, 12, 4, 2, 2, 2, 1.0e-5F, 10000.0F};
    CudaDecoderLayer layer;
    CudaKvCache cache;
    CHECK(CudaDecoderLayer::create(config, &layer).is_ok());
    CHECK(CudaKvCache::create(4, 2, 2, &cache).is_ok());

    std::vector<CudaTensor> tensors(9);
    const std::vector<std::vector<std::size_t>> shapes{
        {8}, {8,8}, {8,4}, {8,4}, {8,8}, {8}, {8,12}, {8,12}, {12,8}};
    std::vector<std::vector<float>> host;
    for (const auto& shape : shapes) {
        std::size_t count = 1;
        for (auto dim : shape) count *= dim;
        host.emplace_back(count, 0.0F);
    }
    host[0].assign(8, 1.0F);
    host[5].assign(8, 1.0F);
    for (std::size_t i = 0; i < 8; ++i) {
        host[1][i * 8 + i] = 1.0F;
        host[4][i * 8 + i] = 1.0F;
        host[6][i * 12 + i] = 0.5F;
        host[7][i * 12 + i] = 0.25F;
        host[8][i * 8 + i] = 0.75F;
    }
    for (std::size_t i = 0; i < 4; ++i) {
        host[2][i * 4 + i] = 1.0F;
        host[3][i * 4 + i] = 1.0F;
    }
    for (std::size_t i = 0; i < tensors.size(); ++i) {
        CHECK(CudaTensor::create_contiguous(shapes[i], DType::Float32, &tensors[i]).is_ok());
        CHECK(context.upload(&tensors[i], host[i].data(), host[i].size() * sizeof(float)).is_ok());
    }
    const CudaDecoderLayerWeights weights{
        &tensors[0], CudaLinearWeight::float32(&tensors[1]),
        CudaLinearWeight::float32(&tensors[2]), CudaLinearWeight::float32(&tensors[3]),
        CudaLinearWeight::float32(&tensors[4]), &tensors[5],
        CudaLinearWeight::float32(&tensors[6]), CudaLinearWeight::float32(&tensors[7]),
        CudaLinearWeight::float32(&tensors[8])};
    std::vector<float> hidden_host{0.2F,-0.4F,0.6F,-0.8F,1.0F,-1.2F,1.4F,-1.6F};
    CudaTensor hidden;
    CHECK(CudaTensor::create_contiguous({8}, DType::Float32, &hidden).is_ok());
    CHECK(context.upload(&hidden, hidden_host.data(), 8 * sizeof(float)).is_ok());
    CHECK(layer.forward(&context, weights, &cache, &hidden, 0).is_ok());

    const std::vector<float> norm1 = rms(hidden_host);
    std::vector<float> attention{norm1[0], norm1[1], norm1[0], norm1[1],
                                 norm1[2], norm1[3], norm1[2], norm1[3]};
    std::vector<float> after_attention(8);
    for (std::size_t i = 0; i < 8; ++i) after_attention[i] = hidden_host[i] + attention[i];
    const std::vector<float> norm2 = rms(after_attention);
    std::vector<float> swiglu(12, 0.0F);
    for (std::size_t i = 0; i < 8; ++i) {
        const float gate = norm2[i] * 0.5F;
        swiglu[i] = gate / (1.0F + std::exp(-gate)) * (norm2[i] * 0.25F);
    }
    std::vector<float> expected(8);
    for (std::size_t i = 0; i < 8; ++i) expected[i] = after_attention[i] + swiglu[i] * 0.75F;
    const auto check_tensor = [&](const CudaTensor& tensor, const std::vector<float>& golden) {
        std::vector<float> actual(golden.size());
        return context.download(tensor, actual.data(), actual.size() * sizeof(float)).is_ok() &&
               context.synchronize().is_ok() && close(actual, golden);
    };
    CHECK(check_tensor(layer.attention_normalized(), norm1));
    CHECK(check_tensor(layer.attention_context(), attention));
    CHECK(check_tensor(layer.ffn_normalized(), norm2));
    CHECK(check_tensor(layer.swiglu_output(), swiglu));
    CHECK(check_tensor(hidden, expected));
    CHECK(cache.size_tokens() == 1);
    CHECK(!layer.forward(&context, weights, &cache, &hidden, 2).is_ok());
    CHECK(cache.size_tokens() == 1);
    CHECK(check_tensor(hidden, expected));

    CudaTensor wrong_down;
    CHECK(CudaTensor::create_contiguous(
        {11,8}, DType::Float32, &wrong_down).is_ok());
    CudaDecoderLayerWeights wrong_weights = weights;
    wrong_weights.down = CudaLinearWeight::float32(&wrong_down);
    CHECK(!layer.forward(&context, wrong_weights, &cache, &hidden, 1).is_ok());
    CHECK(cache.size_tokens() == 1);
    CHECK(check_tensor(hidden, expected));
    CHECK(!layer.forward(nullptr, weights, &cache, &hidden, 1).is_ok());

    CudaExecutionContext sequential_context;
    CudaExecutionContext batch_context;
    CudaDecoderLayer sequential_a;
    CudaDecoderLayer sequential_b;
    CudaKvCache sequential_cache_a;
    CudaKvCache sequential_cache_b;
    CudaKvCache batch_cache_a;
    CudaKvCache batch_cache_b;
    CHECK(CudaExecutionContext::create(&sequential_context).is_ok());
    CHECK(CudaExecutionContext::create(&batch_context).is_ok());
    CHECK(CudaDecoderLayer::create(config, &sequential_a).is_ok());
    CHECK(CudaDecoderLayer::create(config, &sequential_b).is_ok());
    for (CudaKvCache* item : {&sequential_cache_a, &sequential_cache_b,
                              &batch_cache_a, &batch_cache_b}) {
        CHECK(CudaKvCache::create(4, 2, 2, item).is_ok());
    }
    const std::vector<float> hidden_b_host{
        -0.3F, 0.5F, -0.7F, 0.9F, -1.1F, 1.3F, -1.5F, 1.7F};
    CudaTensor sequential_hidden_a;
    CudaTensor sequential_hidden_b;
    CHECK(CudaTensor::create_contiguous(
        {8}, DType::Float32, &sequential_hidden_a).is_ok());
    CHECK(CudaTensor::create_contiguous(
        {8}, DType::Float32, &sequential_hidden_b).is_ok());
    CHECK(sequential_context.upload(
        &sequential_hidden_a, hidden_host.data(), 8 * sizeof(float)).is_ok());
    CHECK(sequential_context.upload(
        &sequential_hidden_b, hidden_b_host.data(), 8 * sizeof(float)).is_ok());
    CHECK(sequential_a.forward(
        &sequential_context, weights, &sequential_cache_a,
        &sequential_hidden_a, 0).is_ok());
    CHECK(sequential_b.forward(
        &sequential_context, weights, &sequential_cache_b,
        &sequential_hidden_b, 0).is_ok());
    std::vector<float> sequential_output_a(8);
    std::vector<float> sequential_output_b(8);
    CHECK(sequential_context.download(
        sequential_hidden_a, sequential_output_a.data(), 8 * sizeof(float)).is_ok());
    CHECK(sequential_context.download(
        sequential_hidden_b, sequential_output_b.data(), 8 * sizeof(float)).is_ok());
    CHECK(sequential_context.synchronize().is_ok());

    CudaDecoderBatchLayer batch_layer;
    CudaTensor batch_hidden;
    CHECK(CudaDecoderBatchLayer::create(config, 2, &batch_layer).is_ok());
    CHECK(CudaTensor::create_contiguous(
        {2, 8}, DType::Float32, &batch_hidden).is_ok());
    std::vector<float> batch_hidden_host = hidden_host;
    batch_hidden_host.insert(
        batch_hidden_host.end(), hidden_b_host.begin(), hidden_b_host.end());
    CHECK(batch_context.upload(
        &batch_hidden,
        batch_hidden_host.data(),
        batch_hidden_host.size() * sizeof(float)).is_ok());
    CHECK(batch_layer.forward(
        &batch_context, weights, {&batch_cache_a, &batch_cache_b},
        &batch_hidden, {0, 0}).is_ok());
    std::vector<float> batch_output(16);
    CHECK(batch_context.download(
        batch_hidden, batch_output.data(), batch_output.size() * sizeof(float)).is_ok());
    CHECK(batch_context.synchronize().is_ok());
    CHECK(close(
        std::vector<float>(batch_output.begin(), batch_output.begin() + 8),
        sequential_output_a));
    CHECK(close(
        std::vector<float>(batch_output.begin() + 8, batch_output.end()),
        sequential_output_b));
    CHECK(batch_cache_a.size_tokens() == 1);
    CHECK(batch_cache_b.size_tokens() == 1);

    CudaDecoderLayerWeights invalid_batch_weights = weights;
    invalid_batch_weights.down = CudaLinearWeight::float32(&wrong_down);
    CHECK(!batch_layer.forward(
        &batch_context, invalid_batch_weights,
        {&batch_cache_a, &batch_cache_b}, &batch_hidden, {1, 1}).is_ok());
    CHECK(batch_context.synchronize().is_ok());
    CHECK(batch_cache_a.size_tokens() == 1);
    CHECK(batch_cache_b.size_tokens() == 1);
    std::vector<float> batch_after_failure(16);
    CHECK(batch_context.download(
        batch_hidden,
        batch_after_failure.data(),
        batch_after_failure.size() * sizeof(float)).is_ok());
    CHECK(batch_context.synchronize().is_ok());
    CHECK(close(batch_after_failure, batch_output));

    CudaExecutionContext prefill_context;
    CudaExecutionContext prefill_sequential_context;
    CudaDecoderBatchLayer prefill_layer;
    CudaDecoderLayer prefill_sequential_layer;
    CudaKvCache prefill_cache;
    CudaKvCache prefill_sequential_cache;
    CHECK(CudaExecutionContext::create(&prefill_context).is_ok());
    CHECK(CudaExecutionContext::create(&prefill_sequential_context).is_ok());
    CHECK(CudaDecoderBatchLayer::create(config, 2, &prefill_layer).is_ok());
    CHECK(CudaDecoderLayer::create(config, &prefill_sequential_layer).is_ok());
    CHECK(CudaKvCache::create(4, 2, 2, &prefill_cache).is_ok());
    CHECK(CudaKvCache::create(4, 2, 2, &prefill_sequential_cache).is_ok());
    CudaTensor prefill_hidden;
    CHECK(CudaTensor::create_contiguous({2, 8}, DType::Float32, &prefill_hidden).is_ok());
    CHECK(prefill_context.upload(
        &prefill_hidden, batch_hidden_host.data(),
        batch_hidden_host.size() * sizeof(float)).is_ok());
    CHECK(prefill_layer.prefill_sequence(
        &prefill_context, weights, &prefill_cache, &prefill_hidden, 0).is_ok());
    std::vector<float> prefill_output(16);
    CHECK(prefill_context.download(
        prefill_hidden, prefill_output.data(), prefill_output.size() * sizeof(float)).is_ok());
    CHECK(prefill_context.synchronize().is_ok());

    std::vector<float> sequential_prefill_output;
    for (std::size_t token = 0; token < 2; ++token) {
        CudaTensor token_hidden;
        CHECK(CudaTensor::create_contiguous({8}, DType::Float32, &token_hidden).is_ok());
        const float* token_input = batch_hidden_host.data() + token * 8;
        CHECK(prefill_sequential_context.upload(
            &token_hidden, token_input, 8 * sizeof(float)).is_ok());
        CHECK(prefill_sequential_layer.forward(
            &prefill_sequential_context, weights, &prefill_sequential_cache,
            &token_hidden, token).is_ok());
        std::vector<float> token_output(8);
        CHECK(prefill_sequential_context.download(
            token_hidden, token_output.data(), 8 * sizeof(float)).is_ok());
        CHECK(prefill_sequential_context.synchronize().is_ok());
        sequential_prefill_output.insert(
            sequential_prefill_output.end(), token_output.begin(), token_output.end());
    }
    CHECK(close(prefill_output, sequential_prefill_output));
    CHECK(prefill_cache.size_tokens() == 2);
    CHECK(prefill_sequential_cache.size_tokens() == 2);
    CHECK(!prefill_layer.prefill_sequence(
        &prefill_context, invalid_batch_weights,
        &prefill_cache, &prefill_hidden, 2).is_ok());
    CHECK(prefill_context.synchronize().is_ok());
    CHECK(prefill_cache.size_tokens() == 2);
    std::vector<float> prefill_after_failure(16);
    CHECK(prefill_context.download(
        prefill_hidden, prefill_after_failure.data(),
        prefill_after_failure.size() * sizeof(float)).is_ok());
    CHECK(prefill_context.synchronize().is_ok());
    CHECK(close(prefill_after_failure, prefill_output));

    CudaExecutionContext paged_prefill_context;
    CudaDecoderBatchLayer paged_prefill_layer;
    CudaPagedKvCache paged_prefill_cache;
    CudaTensor paged_prefill_hidden;
    CHECK(CudaExecutionContext::create(&paged_prefill_context).is_ok());
    CHECK(CudaDecoderBatchLayer::create(config, 2, &paged_prefill_layer).is_ok());
    CHECK(CudaPagedKvCache::create(1, 2, 1, 2, 2, &paged_prefill_cache).is_ok());
    CHECK(CudaTensor::create_contiguous(
        {2, 8}, DType::Float32, &paged_prefill_hidden).is_ok());
    CHECK(paged_prefill_context.upload(
        &paged_prefill_hidden, batch_hidden_host.data(),
        batch_hidden_host.size() * sizeof(float)).is_ok());
    const std::vector<std::size_t> noncontiguous_blocks{1, 0};
    CHECK(paged_prefill_layer.prefill_sequence_paged(
        &paged_prefill_context, weights, &paged_prefill_cache, 0,
        noncontiguous_blocks, &paged_prefill_hidden, 0).is_ok());
    std::vector<float> paged_prefill_output(16);
    CHECK(paged_prefill_context.download(
        paged_prefill_hidden, paged_prefill_output.data(),
        paged_prefill_output.size() * sizeof(float)).is_ok());
    CHECK(paged_prefill_context.synchronize().is_ok());
    CHECK(close(paged_prefill_output, sequential_prefill_output));

    CHECK(!paged_prefill_layer.prefill_sequence_paged(
        &paged_prefill_context, weights, &paged_prefill_cache, 0,
        {2, 0}, &paged_prefill_hidden, 0).is_ok());
    CHECK(!paged_prefill_layer.prefill_sequence_paged(
        &paged_prefill_context, invalid_batch_weights, &paged_prefill_cache, 0,
        noncontiguous_blocks, &paged_prefill_hidden, 0).is_ok());
    CHECK(paged_prefill_context.synchronize().is_ok());
    std::vector<float> paged_after_failure(16);
    CHECK(paged_prefill_context.download(
        paged_prefill_hidden, paged_after_failure.data(),
        paged_after_failure.size() * sizeof(float)).is_ok());
    CHECK(paged_prefill_context.synchronize().is_ok());
    CHECK(close(paged_after_failure, paged_prefill_output));

    // Qwen 类布局让 Q 宽度大于 hidden，并由输出投影收回 hidden。
    CudaExecutionContext asymmetric_context;
    CudaDecoderLayer asymmetric_layer;
    CudaKvCache asymmetric_cache;
    const CudaDecoderLayerConfig asymmetric_config{
        4, 6, 2, 1, 4, 4, 1.0e-5F, 1000000.0F,
        cpp_llm_engine::core::RotaryPairing::SplitHalf};
    CHECK(CudaExecutionContext::create(&asymmetric_context).is_ok());
    CHECK(CudaDecoderLayer::create(asymmetric_config, &asymmetric_layer).is_ok());
    CHECK(CudaKvCache::create(2, 1, 4, &asymmetric_cache).is_ok());
    std::vector<CudaTensor> asymmetric_tensors(11);
    const std::vector<std::vector<std::size_t>> asymmetric_shapes{
        {4}, {4,8}, {4,4}, {4,4}, {8,4}, {4}, {4,6}, {4,6}, {6,4},
        {4}, {4}};
    std::vector<std::vector<float>> asymmetric_host;
    for (const auto& shape : asymmetric_shapes) {
        std::size_t count = 1;
        for (const std::size_t dimension : shape) count *= dimension;
        asymmetric_host.emplace_back(count, 0.0F);
    }
    asymmetric_host[0].assign(4, 1.0F);
    asymmetric_host[5].assign(4, 1.0F);
    asymmetric_host[9].assign(4, 1.0F);
    asymmetric_host[10].assign(4, 1.0F);
    for (std::size_t index = 0; index < 4; ++index) {
        asymmetric_host[1][index * 8 + index] = 1.0F;
        asymmetric_host[1][index * 8 + 4 + index] = 1.0F;
        asymmetric_host[2][index * 4 + index] = 1.0F;
        asymmetric_host[3][index * 4 + index] = 1.0F;
        asymmetric_host[4][index * 4 + index] = 0.5F;
        asymmetric_host[4][(4 + index) * 4 + index] = 0.5F;
    }
    for (std::size_t index = 0; index < asymmetric_tensors.size(); ++index) {
        CHECK(CudaTensor::create_contiguous(
            asymmetric_shapes[index], DType::Float32,
            &asymmetric_tensors[index]).is_ok());
        CHECK(asymmetric_context.upload(
            &asymmetric_tensors[index], asymmetric_host[index].data(),
            asymmetric_host[index].size() * sizeof(float)).is_ok());
    }
    const CudaDecoderLayerWeights asymmetric_weights{
        &asymmetric_tensors[0], CudaLinearWeight::float32(&asymmetric_tensors[1]),
        CudaLinearWeight::float32(&asymmetric_tensors[2]),
        CudaLinearWeight::float32(&asymmetric_tensors[3]),
        CudaLinearWeight::float32(&asymmetric_tensors[4]), &asymmetric_tensors[5],
        CudaLinearWeight::float32(&asymmetric_tensors[6]),
        CudaLinearWeight::float32(&asymmetric_tensors[7]),
        CudaLinearWeight::float32(&asymmetric_tensors[8]),
        &asymmetric_tensors[9], &asymmetric_tensors[10]};
    const std::vector<float> asymmetric_input{0.25F, -0.5F, 0.75F, -1.0F};
    CudaTensor asymmetric_hidden;
    CHECK(CudaTensor::create_contiguous(
        {4}, DType::Float32, &asymmetric_hidden).is_ok());
    CHECK(asymmetric_context.upload(
        &asymmetric_hidden, asymmetric_input.data(),
        asymmetric_input.size() * sizeof(float)).is_ok());
    CHECK(asymmetric_layer.forward(
        &asymmetric_context, asymmetric_weights, &asymmetric_cache,
        &asymmetric_hidden, 0).is_ok());
    const std::vector<float> asymmetric_norm = rms(asymmetric_input);
    std::vector<float> asymmetric_expected(4);
    for (std::size_t index = 0; index < 4; ++index) {
        asymmetric_expected[index] = asymmetric_input[index] + asymmetric_norm[index];
    }
    std::vector<float> asymmetric_output(4);
    CHECK(asymmetric_context.download(
        asymmetric_hidden, asymmetric_output.data(),
        asymmetric_output.size() * sizeof(float)).is_ok());
    CHECK(asymmetric_context.synchronize().is_ok());
    CHECK(close(asymmetric_output, asymmetric_expected));
    CHECK(asymmetric_cache.size_tokens() == 1);
    return 0;
}

#undef CHECK
