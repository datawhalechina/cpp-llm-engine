// Engine CUDA：验证真实 TinyLlama CUDA Executor 接入 Chapter06 Runtime。
// 职责：执行 prefill/decode、双请求 batch、分页 K/V 与 Radix 前缀采用。
// 通过条件：状态按模型调用提交，批量与前缀命中路径均与顺序冷启动一致。
#include "cpp_llm_engine/real_llama_cuda_executor_factory.h"

#include "chapter06_generation/request.h"
#include "chapter06_generation/runtime.h"
#include "chapter06_generation/sampler.h"
#include "chapter07_kv_cache/block_table.h"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

#define CHECK(c) do { if (!(c)) { std::cerr << "check failed line " << __LINE__ << '\n'; return 1; } } while (false)

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "usage: real_tinyllama_cuda_runtime_test <model.gguf>\n";
        return 2;
    }
    int devices = 0;
    CHECK(cudaGetDeviceCount(&devices) == cudaSuccess && devices > 0);

    std::string error;
    auto factory = cpp_llm_engine::RealLlamaCudaExecutorFactory::load(
        argv[1], 8, &error);
    CHECK(factory != nullptr);
    CHECK(factory->layer_count() == 22);

    chapter06_generation::GenerationConfig config;
    config.max_new_tokens = 2;
    config.eos_token_id = factory->eos_token_id();
    chapter06_generation::Runtime runtime(config);
    chapter06_generation::GreedySampler sampler;
    const chapter06_generation::TokenSequence prompt{15043};
    const auto max_error = [](const auto& actual, const auto& expected) {
        float result = 0.0F;
        for (std::size_t index = 0; index < actual.size(); ++index) {
            result = std::max(result, std::fabs(actual[index] - expected[index]));
        }
        return result;
    };

    CHECK(factory->attention_mode() == cpp_llm_engine::CudaAttentionMode::ThreeStage);
    CHECK(factory->quantized_gemv_mode() ==
          cpp_llm_engine::CudaQuantizedGemvMode::Adaptive);
    factory->set_quantized_gemv_mode(
        cpp_llm_engine::CudaQuantizedGemvMode::Adaptive);
    CHECK(factory->quantized_gemv_mode() ==
          cpp_llm_engine::CudaQuantizedGemvMode::Adaptive);
    factory->set_quantized_gemv_mode(
        cpp_llm_engine::CudaQuantizedGemvMode::FixedTile4);
    CHECK(factory->quantized_gemv_mode() ==
          cpp_llm_engine::CudaQuantizedGemvMode::FixedTile4);
    factory->set_quantized_gemv_mode(
        cpp_llm_engine::CudaQuantizedGemvMode::Adaptive);
    auto attention_baseline = factory->create_executor();
    factory->set_attention_mode(cpp_llm_engine::CudaAttentionMode::FusedOnline);
    auto attention_fused = factory->create_executor();
    CHECK(attention_baseline != nullptr && attention_fused != nullptr);
    chapter06_generation::Logits baseline_prefill;
    chapter06_generation::Logits fused_prefill;
    CHECK(attention_baseline->prefill({1, 15043}, &baseline_prefill, &error));
    CHECK(attention_fused->prefill({1, 15043}, &fused_prefill, &error));
    const float attention_prefill_error = max_error(fused_prefill, baseline_prefill);
    CHECK(attention_prefill_error <= 1.0e-3F);
    const auto attention_next_token = sampler.sample(baseline_prefill);
    CHECK(sampler.sample(fused_prefill) == attention_next_token);
    chapter06_generation::Logits baseline_decode;
    chapter06_generation::Logits fused_decode;
    CHECK(attention_baseline->decode(
        attention_next_token, &baseline_decode, &error));
    CHECK(attention_fused->decode(
        attention_next_token, &fused_decode, &error));
    const float attention_decode_error = max_error(fused_decode, baseline_decode);
    CHECK(attention_decode_error <= 1.0e-3F);
    CHECK(sampler.sample(fused_decode) == sampler.sample(baseline_decode));
    factory->set_attention_mode(cpp_llm_engine::CudaAttentionMode::ThreeStage);

    const auto generate = [&](chapter06_generation::TokenSequence* output) {
        auto executor = factory->create_executor();
        if (executor == nullptr || output == nullptr) return false;
        chapter06_generation::Request request(prompt);
        const std::size_t generated = runtime.generate(request, sampler, *executor);
        if (request.failed() || !request.finished() || generated != 2 ||
            request.generated().size() != 2) {
            return false;
        }
        const auto state = executor->execution_state();
        if (!state.available || state.committed_tokens != 2 ||
            state.kv_layer_count != 22 || !state.stores_kv_values) {
            return false;
        }
        *output = request.generated();
        return true;
    };

    chapter06_generation::TokenSequence first;
    chapter06_generation::TokenSequence second;
    CHECK(generate(&first));
    CHECK(generate(&second));
    CHECK(first == second);
    CHECK(first == chapter06_generation::TokenSequence({3561, 12053}));

    auto batch_a = factory->create_executor();
    auto batch_b = factory->create_executor();
    auto sequential_a = factory->create_executor();
    auto sequential_b = factory->create_executor();
    CHECK(batch_a != nullptr && batch_b != nullptr);
    CHECK(sequential_a != nullptr && sequential_b != nullptr);
    chapter06_generation::Logits prefill_a;
    chapter06_generation::Logits prefill_b;
    chapter06_generation::Logits unused;
    CHECK(batch_a->prefill({15043}, &prefill_a, &error));
    CHECK(batch_b->prefill({1, 15043}, &prefill_b, &error));
    CHECK(sequential_a->prefill({15043}, &unused, &error));
    CHECK(sequential_b->prefill({1, 15043}, &unused, &error));
    const chapter06_generation::TokenSequence batch_tokens{
        sampler.sample(prefill_a), sampler.sample(prefill_b)};
    chapter06_generation::Logits expected_a;
    chapter06_generation::Logits expected_b;
    CHECK(sequential_a->decode(batch_tokens[0], &expected_a, &error));
    CHECK(sequential_b->decode(batch_tokens[1], &expected_b, &error));

    std::vector<chapter06_generation::Logits> batch_logits;
    CHECK(!factory->decode_batch(
        {batch_a, batch_b}, batch_tokens, {1, 3}, {101, 102},
        &batch_logits, &error));
    CHECK(batch_a->execution_state().committed_tokens == 1);
    CHECK(batch_b->execution_state().committed_tokens == 2);
    CHECK(factory->decode_batch(
        {batch_a, batch_b}, batch_tokens, {1, 2}, {101, 102},
        &batch_logits, &error));
    CHECK(batch_logits.size() == 2);
    CHECK(batch_logits[0].size() == expected_a.size());
    CHECK(batch_logits[1].size() == expected_b.size());
    const float batch_error_a = max_error(batch_logits[0], expected_a);
    const float batch_error_b = max_error(batch_logits[1], expected_b);
    CHECK(batch_error_a <= 1.0e-3F);
    CHECK(batch_error_b <= 1.0e-3F);
    CHECK(sampler.sample(batch_logits[0]) == sampler.sample(expected_a));
    CHECK(sampler.sample(batch_logits[1]) == sampler.sample(expected_b));
    CHECK(batch_a->execution_state().committed_tokens == 2);
    CHECK(batch_b->execution_state().committed_tokens == 3);
    CHECK(factory->batch_workspace_count() == 1);
    CHECK(factory->batch_workspace_creation_count() == 1);
    CHECK(factory->batch_workspace_use_count(2) == 1);
    CHECK(factory->batch_workspace_device_bytes(2) > 0);
    const std::size_t first_dynamic_allocations =
        factory->batch_workspace_dynamic_allocation_count(2);

    const chapter06_generation::TokenSequence next_batch_tokens{
        sampler.sample(batch_logits[0]), sampler.sample(batch_logits[1])};
    chapter06_generation::Logits next_expected_a;
    chapter06_generation::Logits next_expected_b;
    CHECK(sequential_a->decode(next_batch_tokens[0], &next_expected_a, &error));
    CHECK(sequential_b->decode(next_batch_tokens[1], &next_expected_b, &error));
    CHECK(factory->decode_batch(
        {batch_a, batch_b}, next_batch_tokens, {2, 3}, {101, 102},
        &batch_logits, &error));
    CHECK(max_error(batch_logits[0], next_expected_a) <= 1.0e-3F);
    CHECK(max_error(batch_logits[1], next_expected_b) <= 1.0e-3F);
    CHECK(factory->batch_workspace_count() == 1);
    CHECK(factory->batch_workspace_creation_count() == 1);
    CHECK(factory->batch_workspace_use_count(2) == 2);
    CHECK(factory->batch_workspace_dynamic_allocation_count(2) ==
          first_dynamic_allocations);
    CHECK(batch_a->execution_state().committed_tokens == 3);
    CHECK(batch_b->execution_state().committed_tokens == 4);

    CHECK(factory->configure_paged_kv_cache(8, 2, &error));
    auto continuous = factory->create_executor();
    auto paged = factory->create_executor();
    CHECK(continuous != nullptr && paged != nullptr);
    CHECK(factory->bind_request(paged, 7001, &error));
    auto paged_resource = factory->paged_kv_resource();
    CHECK(paged_resource != nullptr);
    chapter06_generation::Logits continuous_logits;
    chapter06_generation::Logits paged_logits;
    CHECK(continuous->prefill({15043}, &continuous_logits, &error));
    CHECK(paged->prefill({15043}, &paged_logits, &error));
    CHECK(max_error(paged_logits, continuous_logits) <= 1.0e-3F);
    CHECK(paged_resource->token_count(7001) == 1);
    CHECK(paged_resource->block_count(7001) == 1);
    const auto next_token = sampler.sample(continuous_logits);
    CHECK(continuous->decode(next_token, &continuous_logits, &error));
    CHECK(paged->decode(next_token, &paged_logits, &error));
    CHECK(max_error(paged_logits, continuous_logits) <= 1.0e-3F);
    CHECK(paged_resource->token_count(7001) == 2);
    CHECK(paged->execution_state().committed_tokens == 2);
    factory->release_request(7001);
    CHECK(!paged_resource->contains(7001));

    auto paged_batch_a = factory->create_executor();
    auto paged_batch_b = factory->create_executor();
    auto paged_reference_a = factory->create_executor();
    auto paged_reference_b = factory->create_executor();
    CHECK(paged_batch_a != nullptr && paged_batch_b != nullptr);
    CHECK(paged_reference_a != nullptr && paged_reference_b != nullptr);
    CHECK(factory->bind_request(paged_batch_a, 7002, &error));
    CHECK(factory->bind_request(paged_batch_b, 7003, &error));
    chapter06_generation::Logits paged_prefill_a;
    chapter06_generation::Logits paged_prefill_b;
    CHECK(paged_batch_a->prefill({15043}, &paged_prefill_a, &error));
    CHECK(paged_batch_b->prefill({1, 15043}, &paged_prefill_b, &error));
    CHECK(paged_reference_a->prefill({15043}, &unused, &error));
    CHECK(paged_reference_b->prefill({1, 15043}, &unused, &error));
    const chapter06_generation::TokenSequence paged_batch_tokens{
        sampler.sample(paged_prefill_a), sampler.sample(paged_prefill_b)};
    chapter06_generation::Logits paged_expected_a;
    chapter06_generation::Logits paged_expected_b;
    CHECK(paged_reference_a->decode(
        paged_batch_tokens[0], &paged_expected_a, &error));
    CHECK(paged_reference_b->decode(
        paged_batch_tokens[1], &paged_expected_b, &error));

    const std::vector<std::size_t> paged_positions{1, 2};
    CHECK(paged_resource->reserve(7002, 2));
    CHECK(paged_resource->reserve(7003, 3));
    CHECK(paged_resource->token_count(7002) == 1);
    CHECK(paged_resource->token_count(7003) == 2);
    CHECK(factory->decode_batch(
        {paged_batch_a, paged_batch_b}, paged_batch_tokens,
        paged_positions, {7002, 7003}, &batch_logits, &error));
    CHECK(batch_logits.size() == 2);
    CHECK(max_error(batch_logits[0], paged_expected_a) <= 1.0e-3F);
    CHECK(max_error(batch_logits[1], paged_expected_b) <= 1.0e-3F);
    CHECK(paged_batch_a->execution_state().committed_tokens == 2);
    CHECK(paged_batch_b->execution_state().committed_tokens == 3);
    CHECK(paged_resource->token_count(7002) == 1);
    CHECK(paged_resource->token_count(7003) == 2);
    CHECK(paged_resource->commit_tokens(7002, 2));
    CHECK(paged_resource->commit_tokens(7003, 3));

    CHECK(paged_resource->reserve(7002, 3));
    CHECK(paged_resource->reserve(7003, 4));
    const auto paged_position_a = paged_batch_a->execution_state().committed_tokens;
    const auto paged_position_b = paged_batch_b->execution_state().committed_tokens;
    CHECK(!factory->decode_batch(
        {paged_batch_a, paged_batch_b},
        {static_cast<chapter06_generation::TokenId>(factory->vocab_size()),
         sampler.sample(batch_logits[1])},
        {paged_position_a, paged_position_b}, {7002, 7003},
        &batch_logits, &error));
    CHECK(paged_batch_a->execution_state().committed_tokens == paged_position_a);
    CHECK(paged_batch_b->execution_state().committed_tokens == paged_position_b);
    CHECK(paged_resource->trim_to_tokens(7002, paged_position_a));
    CHECK(paged_resource->trim_to_tokens(7003, paged_position_b));
    factory->release_request(7002);
    factory->release_request(7003);
    CHECK(paged_resource->free_blocks() == paged_resource->total_blocks());

    const chapter06_generation::TokenSequence radix_source_prompt{
        15043, 18439, 1460, 29931, 5288, 29874};
    const chapter06_generation::TokenSequence radix_target_prompt{
        15043, 18439, 1460, 29931, 29871, 29973};
    auto radix_source = factory->create_executor();
    auto radix_target = factory->create_executor();
    auto radix_cold = factory->create_executor();
    CHECK(radix_source != nullptr && radix_target != nullptr && radix_cold != nullptr);
    chapter06_generation::Request radix_source_request(radix_source_prompt);
    chapter06_generation::Request radix_target_request(radix_target_prompt);
    CHECK(factory->bind_request(radix_source, 7101, &error));
    const auto radix_source_result = runtime.prefill_chunk(
        radix_source_request, sampler, *radix_source, radix_source_prompt.size());
    CHECK(radix_source_result.generated_tokens == 1);
    factory->publish_cached_prefix(7101, radix_source_request);
    factory->release_request(7101);
    CHECK(factory->bind_request(radix_target, 7102, &error));
    CHECK(factory->adopt_cached_prefix(
        radix_target, 7102, radix_target_request, &error));
    CHECK(radix_target_request.prefill_offset() == 4);
    CHECK(radix_target->execution_state().committed_tokens == 4);
    const auto radix_target_result = runtime.prefill_chunk(
        radix_target_request, sampler, *radix_target, radix_target_prompt.size());
    chapter06_generation::Logits radix_cold_logits;
    CHECK(radix_cold->prefill(radix_target_prompt, &radix_cold_logits, &error));
    CHECK(radix_target_result.generated_tokens == 1);
    CHECK(radix_target_request.generated().front() ==
          sampler.sample(radix_cold_logits));
    cpp_llm_engine::RealLlamaCudaPrefixCacheStats radix_stats;
    CHECK(factory->prefix_cache_stats(&radix_stats, &error));
    CHECK(radix_stats.hits == 1);
    CHECK(radix_stats.matched_tokens == 4);
    CHECK(radix_stats.cached_blocks == 3);
    factory->release_request(7102);

    std::cout << "cuda_runtime_generation: true\n"
              << "generated_tokens:";
    for (const auto token : first) std::cout << ' ' << token;
    std::cout << "\ncontext_isolation: true\n"
              << "cuda_decode_batch: true\n"
              << "persistent_batch_workspace: true\n"
              << "cuda_paged_context: true\n"
              << "cuda_paged_decode_batch: true\n"
              << "cuda_radix_context: true\n"
              << "batch_workspace_bytes: "
              << factory->batch_workspace_device_bytes(2) << '\n'
              << "batch_max_error: " << std::max(batch_error_a, batch_error_b) << '\n'
              << "attention_prefill_max_error: " << attention_prefill_error << '\n'
              << "attention_decode_max_error: " << attention_decode_error << '\n';
    return 0;
}

#undef CHECK
