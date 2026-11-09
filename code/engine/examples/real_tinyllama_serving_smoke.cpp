// Engine/Chapter 10：真实 Llama/Qwen3 双请求服务 smoke。
// 职责：验证双请求服务、终态释放和真实 Radix 命中调度。
// 运行：real_tinyllama_serving_smoke <model.gguf>。
#include "cpp_llm_engine/real_llama_executor_factory.h"
#include "cpp_llm_engine/real_tokenizer_backend.h"
#include "chapter06_generation/request.h"
#include "chapter06_generation/runtime.h"
#include "chapter06_generation/sampler.h"
#include "chapter07_kv_cache/block_table.h"
#include "chapter10_serving/service_core.h"

#include <iostream>
#include <memory>
#include <string>
#include <utility>

int main(int argc, char** argv) {
    using chapter06_generation::Request;
    using chapter06_generation::TokenSequence;

    if (argc != 2) {
        std::cerr << "usage: real_tinyllama_serving_smoke <model.gguf>\n";
        return 2;
    }

    std::string error;
    auto tokenizer = cpp_llm_engine::RealTokenizerBackend::load(argv[1], &error);
    auto factory = cpp_llm_engine::RealLlamaExecutorFactory::load(argv[1], &error);
    if (tokenizer == nullptr || factory == nullptr) {
        std::cerr << "real service load failed: " << error << '\n';
        return 1;
    }
    if (tokenizer->vocab_size() != factory->vocab_size()) {
        std::cerr << "tokenizer and model vocabulary sizes differ\n";
        return 1;
    }
    const bool qwen3 = factory->architecture() == "qwen3";
    const auto decoder_contract = factory->decoder_contract();
    const bool decoder_contract_match = decoder_contract.validate().is_ok() &&
        decoder_contract.vocab_size == (qwen3 ? 151936 : 32000) &&
        decoder_contract.layer_count == (qwen3 ? 28 : 22) &&
        decoder_contract.layer.hidden_size == (qwen3 ? 1024 : 2048) &&
        decoder_contract.layer.ffn_hidden_size == (qwen3 ? 3072 : 5632) &&
        decoder_contract.layer.query_head_count == (qwen3 ? 16 : 32) &&
        decoder_contract.layer.kv_head_count == (qwen3 ? 8 : 4) &&
        decoder_contract.layer.normalization ==
            cpp_llm_engine::core::DecoderNormalizationKind::RmsNorm &&
        decoder_contract.layer.activation ==
            cpp_llm_engine::core::DecoderActivationKind::SwiGlu &&
        decoder_contract.layer.position_encoding ==
            cpp_llm_engine::core::DecoderPositionEncodingKind::Rotary &&
        decoder_contract.layer.uses_kv_cache;

    chapter06_generation::GenerationConfig config;
    config.max_new_tokens = 4;
    config.eos_token_id = factory->eos_token_id();
    chapter06_generation::Runtime runtime(config);
    chapter06_generation::GreedySampler sampler;
    Request request_a(tokenizer->encode(qwen3 ? "Hello Qwen3" : "Hello TinyLlama"));
    Request request_b(tokenizer->encode("Hi"));

    auto executor_a = factory->create_executor();
    auto executor_b = factory->create_executor();
    std::weak_ptr<chapter06_generation::RequestModelExecutor> weak_a = executor_a;
    std::weak_ptr<chapter06_generation::RequestModelExecutor> weak_b = executor_b;

    if (!factory->configure_paged_kv_cache(8, 4, &error)) {
        std::cerr << "paged KV setup failed: " << error << '\n';
        return 1;
    }
    auto block_pool = factory->paged_kv_resource();
    chapter10_serving::ServiceCore service(
        2, 2, {}, block_pool, {}, factory.get());
    const bool admitted_a = service.admit(
        101, request_a, sampler, std::move(executor_a)).accepted();
    const bool admitted_b = service.admit(
        102, request_b, sampler, std::move(executor_b)).accepted();
    chapter10_serving::DispatchStats stats;
    std::size_t dispatch_steps = 0;
    while ((!request_a.terminal() || !request_b.terminal()) && dispatch_steps < 32) {
        const auto step = service.dispatch_step(runtime, 3);
        stats.generated_tokens += step.generated_tokens;
        stats.model_batch_calls += step.model_batch_calls;
        stats.tensor_batch_requests += step.tensor_batch_requests;
        ++dispatch_steps;
        if (step.scheduled_requests == 0) break;
    }
    stats.completed = static_cast<std::size_t>(request_a.finished()) +
        static_cast<std::size_t>(request_b.finished());

    const TokenSequence expected_a = qwen3
        ? TokenSequence{11, 358, 1366, 311}
        : TokenSequence{29892, 306, 29915, 29885};
    const TokenSequence expected_b = qwen3
        ? TokenSequence{18155, 220, 17, 15}
        : TokenSequence{839, 744, 29892, 263};
    const bool request_isolation_match =
        admitted_a && admitted_b && stats.completed == 2 &&
        stats.generated_tokens == 8 && request_a.generated() == expected_a &&
        request_b.generated() == expected_b && stats.model_batch_calls > 0 &&
        stats.tensor_batch_requests >= 2;
    const bool cache_isolated =
        request_a.history().size() == request_a.prompt_size() + expected_a.size() &&
        request_b.history().size() == request_b.prompt_size() + expected_b.size();
    const bool chunked_prefill_match =
        request_a.prompt_size() > 3 && request_a.prefill_offset() == request_a.prompt_size() &&
        request_b.prefill_offset() == request_b.prompt_size() && dispatch_steps > 1 &&
        block_pool->token_count(101) == request_a.prompt_size() + expected_a.size() - 1 &&
        block_pool->token_count(102) == request_b.prompt_size() + expected_b.size() - 1;
    const bool paged_payload_active =
        factory->executor_uses_paged_kv(weak_a.lock()) &&
        factory->executor_uses_paged_kv(weak_b.lock());

    const std::size_t released = service.release_terminal();
    cpp_llm_engine::RealLlamaPrefixCacheStats retained_stats;
    const bool retained_stats_ready =
        factory->prefix_cache_stats(&retained_stats, &error);
    const bool resources_released = released == 2 &&
        request_a.history().empty() && request_b.history().empty() &&
        weak_a.expired() && weak_b.expired() &&
        !block_pool->contains(101) && !block_pool->contains(102);
    const bool reusable_prefix_retained = retained_stats_ready &&
        retained_stats.cached_blocks > 0 &&
        block_pool->free_blocks() < block_pool->total_blocks();
    const auto& metrics = service.metrics();
    const bool metrics_match =
        metrics.accepted_total == 2 && metrics.completed_total == 2 &&
        metrics.failed_total == 0 && metrics.released_total == 2 &&
        metrics.generated_tokens_total == 8;

    chapter06_generation::GenerationConfig cache_config;
    cache_config.max_new_tokens = 1;
    cache_config.eos_token_id = factory->eos_token_id();
    chapter06_generation::Runtime cache_runtime(cache_config);
    TokenSequence shared_prompt = qwen3
        ? request_a.prompt()
        : TokenSequence{15043, 18439, 1460, 29931, 5288, 29874};
    TokenSequence cached_prompt = qwen3
        ? request_a.prompt()
        : TokenSequence{15043, 18439, 1460, 29931, 29871, 29973};
    if (qwen3) {
        shared_prompt.insert(shared_prompt.end(), {11, 13});
        cached_prompt.insert(cached_prompt.end(), {198, 0});
    }
    Request source(shared_prompt);
    chapter10_serving::ServiceCore source_service(
        1, 1, {}, block_pool, {}, factory.get());
    const bool source_admitted = source_service.admit(
        201, source, sampler, factory->create_executor()).accepted();
    const auto source_step = source_admitted
        ? source_service.dispatch_step(cache_runtime, shared_prompt.size())
        : chapter10_serving::DispatchStats{};
    const bool source_published = source.finished() &&
        source_step.generated_tokens == 1 && source_service.release_terminal() == 1;
    cpp_llm_engine::RealLlamaPrefixCacheStats stats_before_scheduling;
    const bool stats_before_ready =
        factory->prefix_cache_stats(&stats_before_scheduling, &error);

    Request cold({1, shared_prompt.front()});
    Request cached(cached_prompt);
    chapter10_serving::ServiceCore cache_service(
        2, 1, {}, block_pool, {}, factory.get());
    const bool cold_admitted = cache_service.admit(
        202, cold, sampler, factory->create_executor()).accepted();
    const bool cached_admitted = cache_service.admit(
        203, cached, sampler, factory->create_executor()).accepted();
    const bool waiting_requests_not_adopted =
        cold.prefill_offset() == 0 && cached.prefill_offset() == 0;
    std::size_t first_scheduled_id = 0;
    const auto cache_step = cache_service.dispatch_step(
        cache_runtime,
        2,
        [&](std::size_t request_id, chapter06_generation::TokenId, std::size_t) {
            first_scheduled_id = request_id;
        });
    cpp_llm_engine::RealLlamaPrefixCacheStats stats_after_scheduling;
    const bool stats_after_ready =
        factory->prefix_cache_stats(&stats_after_scheduling, &error);
    const bool cached_prefix_adopted_at_promotion = stats_before_ready &&
        stats_after_ready &&
        stats_after_scheduling.hits == stats_before_scheduling.hits + 1 &&
        stats_after_scheduling.matched_tokens >=
            stats_before_scheduling.matched_tokens + 4;
    const bool cache_aware_scheduling = source_published && cold_admitted &&
        cached_admitted && waiting_requests_not_adopted &&
        cached_prefix_adopted_at_promotion && cache_step.generated_tokens == 1 &&
        first_scheduled_id == 203 && cached.finished() && !cold.prefilled();
    if (!cold.terminal()) cache_service.cancel(202);
    if (!cached.terminal()) cache_service.cancel(203);
    const bool cache_requests_released = cache_service.release_terminal() == 2 &&
        !block_pool->contains(202) && !block_pool->contains(203);

    const bool real_serving_match = request_isolation_match && cache_isolated &&
        chunked_prefill_match && paged_payload_active && resources_released &&
        reusable_prefix_retained && metrics_match && decoder_contract_match &&
        cache_aware_scheduling && cache_requests_released;

    std::cout << "architecture: " << factory->architecture() << '\n'
              << "model_layers: " << factory->layer_count() << '\n'
              << "request_a_tokens:";
    for (const auto token : request_a.generated()) std::cout << ' ' << token;
    std::cout << "\nrequest_b_tokens:";
    for (const auto token : request_b.generated()) std::cout << ' ' << token;
    std::cout << '\n'
              << "request_isolation_match: " << std::boolalpha
              << request_isolation_match << '\n'
              << "cache_isolated: " << cache_isolated << '\n'
              << "chunked_prefill_match: " << chunked_prefill_match << '\n'
              << "paged_payload_active: " << paged_payload_active << '\n'
              << "resources_released: " << resources_released << '\n'
              << "reusable_prefix_retained: " << reusable_prefix_retained << '\n'
              << "waiting_requests_not_adopted: "
              << waiting_requests_not_adopted << '\n'
              << "cached_prefix_adopted_at_promotion: "
              << cached_prefix_adopted_at_promotion << '\n'
              << "cache_aware_scheduling: " << cache_aware_scheduling << '\n'
              << "cache_requests_released: " << cache_requests_released << '\n'
              << "metrics_match: " << metrics_match << '\n'
              << "decoder_contract_match: " << decoder_contract_match << '\n'
              << "real_serving_match: " << real_serving_match << '\n'
              << "real_serving: ready\n";

    return real_serving_match ? 0 : 1;
}
