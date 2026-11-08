// Chapter 07：真实 TinyLlama 分页 K/V 数据面 smoke。
// 职责：比较连续 Context 与单请求分页 Context，并验证提交、回滚和释放。
// 验证：real_tinyllama_paged_kv_test 使用固定 prompt 和 greedy token。
#include "chapter06_generation/sampler.h"
#include "chapter06_generation/request.h"
#include "chapter06_generation/runtime.h"
#include "chapter07_kv_cache/paged_kv_cache.h"
#include "cpp_llm_engine/real_llama_executor_factory.h"

#include <cmath>
#include <iostream>
#include <string>
#include <vector>

namespace {

bool logits_match(const chapter06_generation::Logits& left,
                  const chapter06_generation::Logits& right) {
    if (left.size() != right.size()) return false;
    for (std::size_t index = 0; index < left.size(); ++index) {
        if (std::abs(left[index] - right[index]) > 1.0e-6f) return false;
    }
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "usage: real_tinyllama_paged_kv_smoke <model.gguf>\n";
        return 2;
    }

    std::string error;
    auto factory = cpp_llm_engine::RealLlamaExecutorFactory::load(argv[1], &error);
    if (factory == nullptr) {
        std::cerr << "real paged KV load failed: " << error << '\n';
        return 1;
    }

    auto contiguous = factory->create_executor();
    auto paged = factory->create_executor();
    if (contiguous == nullptr || paged == nullptr) {
        std::cerr << "real paged KV executor creation failed\n";
        return 1;
    }

    const chapter06_generation::TokenSequence prompt{
        15043, 18439, 1460, 29931, 5288, 29874};
    chapter06_generation::Logits contiguous_logits;
    chapter06_generation::Logits paged_logits;
    if (!contiguous->prefill(prompt, &contiguous_logits, &error) ||
        !paged->prefill(prompt, &paged_logits, &error)) {
        std::cerr << "real paged KV prefill failed: " << error << '\n';
        return 1;
    }

    const auto contract = factory->decoder_contract();
    chapter07_kv_cache::CpuPagedKvCache cache(
        contract.layer_count,
        8,
        4,
        contract.layer.kv_head_count,
        contract.layer.head_dimension);
    constexpr std::size_t kRequestId = 7001;
    const std::size_t initial_position = paged->execution_state().committed_tokens;
    const bool initial_sync = factory->sync_paged_kv(
        paged, kRequestId, &cache, &error);
    const bool initial_count_match =
        cache.token_count(kRequestId) == initial_position;

    chapter06_generation::GreedySampler sampler;
    const auto contiguous_token = sampler.sample(contiguous_logits);
    const auto paged_token = sampler.sample(paged_logits);
    const bool prefill_match = logits_match(contiguous_logits, paged_logits) &&
        contiguous_token == paged_token && contiguous_token == 29892;

    if (!contiguous->decode(contiguous_token, &contiguous_logits, &error)) {
        std::cerr << "continuous decode failed: " << error << '\n';
        return 1;
    }
    std::vector<chapter06_generation::Logits> paged_batch_logits;
    const bool decode_succeeded = factory->decode_batch_paged(
        {paged},
        {paged_token},
        {initial_position},
        {kRequestId},
        &cache,
        &paged_batch_logits,
        &error);
    const bool decode_match = decode_succeeded && paged_batch_logits.size() == 1 &&
        logits_match(contiguous_logits, paged_batch_logits.front());
    const std::size_t committed_position = initial_position + 1;
    const bool commit_match = paged->execution_state().committed_tokens ==
            committed_position &&
        cache.token_count(kRequestId) == committed_position;

    std::vector<chapter06_generation::Logits> rejected_logits;
    const bool position_rejected = !factory->decode_batch_paged(
        {paged},
        {paged_token},
        {committed_position + 1},
        {kRequestId},
        &cache,
        &rejected_logits,
        &error);
    const auto invalid_token = static_cast<chapter06_generation::TokenId>(
        factory->vocab_size());
    const bool invalid_token_rejected = !factory->decode_batch_paged(
        {paged},
        {invalid_token},
        {committed_position},
        {kRequestId},
        &cache,
        &rejected_logits,
        &error);
    const bool rollback_match = paged->execution_state().committed_tokens ==
            committed_position &&
        cache.token_count(kRequestId) == committed_position;

    const bool released = cache.release(kRequestId);
    const bool release_match = released && !cache.contains(kRequestId) &&
        cache.free_blocks() == cache.total_blocks();
    const chapter06_generation::TokenSequence shared_prompt{
        15043, 18439, 1460, 29931, 5288, 29874};
    const chapter06_generation::TokenSequence suffix_prompt{
        15043, 18439, 1460, 29931, 29871, 29973};
    chapter06_generation::GenerationConfig generation_config;
    generation_config.max_new_tokens = 1;
    generation_config.max_context_tokens = 32;
    chapter06_generation::Runtime runtime(generation_config);
    chapter06_generation::Request source_request(shared_prompt);
    chapter06_generation::Request target_request(suffix_prompt);
    auto source_executor = factory->create_executor();
    auto target_executor = factory->create_executor();
    auto cold_executor = factory->create_executor();
    chapter06_generation::Logits cold_logits;
    const bool radix_configured = factory->configure_paged_kv_cache(3, 2, &error);
    const bool source_bound = radix_configured && factory->bind_request(
        source_executor, 8001, &error);
    const auto source_result = source_bound
        ? runtime.prefill_chunk(
            source_request, sampler, *source_executor, shared_prompt.size())
        : chapter06_generation::PrefillChunkResult{};
    if (source_result.generated_tokens == 1) {
        factory->publish_cached_prefix(8001, source_request);
        factory->release_request(8001);
    }
    const bool target_bound = source_result.generated_tokens == 1 &&
        factory->bind_request(target_executor, 8002, &error);
    const bool target_adopted = target_bound && factory->adopt_cached_prefix(
        target_executor, 8002, target_request, &error);
    const bool adopted_state = target_adopted &&
        target_request.prefill_offset() == 4 &&
        target_executor->execution_state().committed_tokens == 4;
    const auto target_result = adopted_state
        ? runtime.prefill_chunk(
            target_request, sampler, *target_executor, suffix_prompt.size())
        : chapter06_generation::PrefillChunkResult{};
    const bool cold_ready = cold_executor->prefill(suffix_prompt, &cold_logits, &error);
    const bool radix_context_match = target_result.generated_tokens == 1 && cold_ready &&
        target_request.generated().front() == sampler.sample(cold_logits);
    cpp_llm_engine::RealLlamaPrefixCacheStats prefix_stats;
    const bool eviction_match = factory->prefix_cache_stats(&prefix_stats, &error) &&
        prefix_stats.hits == 1 && prefix_stats.misses == 0 &&
        prefix_stats.matched_tokens == 4 && prefix_stats.evicted_blocks == 1;
    factory->release_request(8002);

    const bool passed = initial_sync && initial_count_match && prefill_match &&
        decode_match && commit_match && position_rejected &&
        invalid_token_rejected && rollback_match && release_match &&
        radix_context_match && eviction_match;

    std::cout << "initial_sync: " << std::boolalpha << initial_sync << '\n'
              << "prefill_logits_match: " << prefill_match << '\n'
              << "decode_logits_match: " << decode_match << '\n'
              << "commit_match: " << commit_match << '\n'
              << "position_rejected: " << position_rejected << '\n'
              << "invalid_token_rejected: " << invalid_token_rejected << '\n'
              << "rollback_match: " << rollback_match << '\n'
              << "release_match: " << release_match << '\n'
              << "radix_context_match: " << radix_context_match << '\n'
              << "radix_eviction_match: " << eviction_match << '\n'
              << "paged_kv_real_match: " << passed << '\n';
    return passed ? 0 : 1;
}
