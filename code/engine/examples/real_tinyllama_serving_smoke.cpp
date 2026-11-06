// Engine/Chapter 10：真实 TinyLlama 双请求服务 smoke。
// 职责：验证共享 Model、独立 Context、Step batch 执行和终态资源释放。
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
    const auto decoder_contract = factory->decoder_contract();
    const bool decoder_contract_match = decoder_contract.validate().is_ok() &&
        decoder_contract.vocab_size == 32000 && decoder_contract.layer_count == 22 &&
        decoder_contract.layer.hidden_size == 2048 &&
        decoder_contract.layer.ffn_hidden_size == 5632 &&
        decoder_contract.layer.query_head_count == 32 &&
        decoder_contract.layer.kv_head_count == 4 &&
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
    Request request_a(tokenizer->encode("Hello TinyLlama"));
    Request request_b(tokenizer->encode("Hi"));

    auto executor_a = factory->create_executor();
    auto executor_b = factory->create_executor();
    std::weak_ptr<chapter06_generation::RequestModelExecutor> weak_a = executor_a;
    std::weak_ptr<chapter06_generation::RequestModelExecutor> weak_b = executor_b;

    if (!factory->configure_paged_kv_cache(8, 4, &error)) {
        std::cerr << "paged KV setup failed: " << error << '\n';
        return 1;
    }
    auto block_pool = factory->paged_block_pool();
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

    const TokenSequence expected_a{29892, 306, 29915, 29885};
    const TokenSequence expected_b{839, 744, 29892, 263};
    const bool request_isolation_match =
        admitted_a && admitted_b && stats.completed == 2 &&
        stats.generated_tokens == 8 && request_a.generated() == expected_a &&
        request_b.generated() == expected_b && stats.model_batch_calls > 0 &&
        stats.tensor_batch_requests >= 2;
    const bool cache_isolated =
        request_a.cache().size() == request_a.prompt_size() + expected_a.size() &&
        request_b.cache().size() == request_b.prompt_size() + expected_b.size();
    const bool chunked_prefill_match =
        request_a.prompt_size() > 3 && request_a.prefill_offset() == request_a.prompt_size() &&
        request_b.prefill_offset() == request_b.prompt_size() && dispatch_steps > 1 &&
        block_pool->token_count(101) == request_a.prompt_size() + expected_a.size() - 1 &&
        block_pool->token_count(102) == request_b.prompt_size() + expected_b.size() - 1;
    const bool paged_payload_active =
        factory->executor_uses_paged_kv(weak_a.lock()) &&
        factory->executor_uses_paged_kv(weak_b.lock());

    const std::size_t released = service.release_terminal();
    const bool resources_released =
        released == 2 && request_a.cache().empty() && request_b.cache().empty() &&
        weak_a.expired() && weak_b.expired() && block_pool->free_blocks() == 8;
    const auto& metrics = service.metrics();
    const bool metrics_match =
        metrics.accepted_total == 2 && metrics.completed_total == 2 &&
        metrics.failed_total == 0 && metrics.released_total == 2 &&
        metrics.generated_tokens_total == 8;

    std::cout << "model_layers: " << factory->layer_count() << '\n'
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
              << "metrics_match: " << metrics_match << '\n'
              << "decoder_contract_match: " << decoder_contract_match << '\n'
              << "real_serving: ready\n";

    return request_isolation_match && cache_isolated && chunked_prefill_match &&
        paged_payload_active &&
        resources_released && metrics_match && decoder_contract_match ? 0 : 1;
}
