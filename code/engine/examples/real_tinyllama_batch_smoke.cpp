// Engine：真实 TinyLlama 按层交错 batch smoke。
// 职责：比较顺序 decode 与共享 Model 的多 Context 分层推进结果。
// 验证：不同长度 prompt 的 logits、golden token、position 拒绝和状态提交。
#include "chapter06_generation/sampler.h"
#include "chapter07_kv_cache/paged_kv_cache.h"
#include "cpp_llm_engine/real_llama_executor_factory.h"
#include "cpp_llm_engine/real_tokenizer_backend.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace {

using Executor = chapter06_generation::RequestModelExecutor;
using ExecutorList = std::vector<std::shared_ptr<Executor>>;
using LogitsBatch = std::vector<chapter06_generation::Logits>;
using TokenSequence = chapter06_generation::TokenSequence;

bool logits_match(const chapter06_generation::Logits& left,
                  const chapter06_generation::Logits& right) {
    if (left.size() != right.size()) return false;
    for (std::size_t index = 0; index < left.size(); ++index) {
        if (std::abs(left[index] - right[index]) > 1.0e-6f) return false;
    }
    return true;
}

bool prefill_all(const ExecutorList& executors,
                 const std::vector<TokenSequence>& prompts,
                 LogitsBatch* logits,
                 std::string* error) {
    logits->resize(executors.size());
    for (std::size_t index = 0; index < executors.size(); ++index) {
        if (!executors[index]->prefill(prompts[index], &(*logits)[index], error)) {
            return false;
        }
    }
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "usage: real_tinyllama_batch_smoke <model.gguf>\n";
        return 2;
    }

    std::string error;
    auto tokenizer = cpp_llm_engine::RealTokenizerBackend::load(argv[1], &error);
    auto factory = cpp_llm_engine::RealLlamaExecutorFactory::load(argv[1], &error);
    if (tokenizer == nullptr || factory == nullptr) {
        std::cerr << "real batch load failed: " << error << '\n';
        return 1;
    }

    const std::vector<TokenSequence> prompts{
        tokenizer->encode("Hello TinyLlama"),
        tokenizer->encode("Hi"),
    };
    ExecutorList sequential{factory->create_executor(), factory->create_executor()};
    ExecutorList batched{factory->create_executor(), factory->create_executor()};
    LogitsBatch sequential_logits;
    LogitsBatch batch_logits;
    if (!prefill_all(sequential, prompts, &sequential_logits, &error) ||
        !prefill_all(batched, prompts, &batch_logits, &error)) {
        std::cerr << "real batch prefill failed: " << error << '\n';
        return 1;
    }

    const std::vector<std::size_t> initial_positions{
        batched[0]->execution_state().committed_tokens,
        batched[1]->execution_state().committed_tokens,
    };
    const auto contract = factory->decoder_contract();
    chapter07_kv_cache::CpuPagedKvCache paged_cache(
        contract.layer_count,
        16,
        4,
        contract.layer.kv_head_count,
        contract.layer.head_dimension);
    const std::vector<std::size_t> request_ids{100, 101};
    bool paged_kv_match = true;
    for (std::size_t index = 0; index < batched.size(); ++index) {
        paged_kv_match = paged_kv_match && factory->sync_paged_kv(
            batched[index], request_ids[index], &paged_cache, &error);
        paged_kv_match = paged_kv_match &&
            paged_cache.block_pool().token_count(request_ids[index]) ==
                initial_positions[index];
    }
    if (!paged_kv_match) {
        std::cerr << "paged KV initial sync failed: " << error << '\n';
        return 1;
    }
    LogitsBatch rejected_logits;
    const bool position_rejected = !factory->decode_batch_paged(
        batched,
        {1, 1},
        {initial_positions[0] + 1, initial_positions[1]},
        request_ids,
        &paged_cache,
        &rejected_logits,
        &error);
    const bool rejection_preserved_state =
        batched[0]->execution_state().committed_tokens == initial_positions[0] &&
        batched[1]->execution_state().committed_tokens == initial_positions[1];
    const auto invalid_token = static_cast<chapter06_generation::TokenId>(
        factory->vocab_size());
    const bool invalid_token_rejected = !factory->decode_batch_paged(
        batched,
        {1, invalid_token},
        initial_positions,
        request_ids,
        &paged_cache,
        &rejected_logits,
        &error);
    const bool rollback_preserved_state =
        batched[0]->execution_state().committed_tokens == initial_positions[0] &&
        batched[1]->execution_state().committed_tokens == initial_positions[1] &&
        paged_cache.block_pool().token_count(request_ids[0]) == initial_positions[0] &&
        paged_cache.block_pool().token_count(request_ids[1]) == initial_positions[1];

    chapter06_generation::GreedySampler sampler;
    std::vector<TokenSequence> sequential_tokens(2);
    std::vector<TokenSequence> batch_tokens(2);
    bool logits_equal = logits_match(sequential_logits[0], batch_logits[0]) &&
        logits_match(sequential_logits[1], batch_logits[1]);
    bool states_committed = true;

    constexpr std::size_t kNewTokens = 4;
    for (std::size_t step = 0; step < kNewTokens; ++step) {
        TokenSequence next_tokens(2);
        for (std::size_t index = 0; index < next_tokens.size(); ++index) {
            const auto sequential_token = sampler.sample(sequential_logits[index]);
            const auto batch_token = sampler.sample(batch_logits[index]);
            sequential_tokens[index].push_back(sequential_token);
            batch_tokens[index].push_back(batch_token);
            next_tokens[index] = batch_token;
            logits_equal = logits_equal && sequential_token == batch_token;
        }
        if (step + 1 == kNewTokens) break;

        std::vector<std::size_t> expected_positions(2);
        for (std::size_t index = 0; index < sequential.size(); ++index) {
            if (!sequential[index]->decode(
                    next_tokens[index], &sequential_logits[index], &error)) {
                std::cerr << "sequential decode failed: " << error << '\n';
                return 1;
            }
            expected_positions[index] = batched[index]->execution_state().committed_tokens;
        }
        if (!factory->decode_batch_paged(
                batched,
                next_tokens,
                expected_positions,
                request_ids,
                &paged_cache,
                &batch_logits,
                &error)) {
            std::cerr << "batch decode failed: " << error << '\n';
            return 1;
        }
        for (std::size_t index = 0; index < batch_logits.size(); ++index) {
            logits_equal = logits_equal &&
                logits_match(sequential_logits[index], batch_logits[index]);
            states_committed = states_committed &&
                batched[index]->execution_state().committed_tokens ==
                    expected_positions[index] + 1;
            paged_kv_match = paged_kv_match &&
                paged_cache.block_pool().token_count(request_ids[index]) ==
                    expected_positions[index] + 1;
        }
    }

    const std::vector<TokenSequence> expected{
        {29892, 306, 29915, 29885},
        {839, 744, 29892, 263},
    };
    const bool golden_match = sequential_tokens == expected && batch_tokens == expected;
    const bool batch_match = sequential_tokens == batch_tokens && logits_equal;
    std::cout << "position_rejected: " << std::boolalpha << position_rejected << '\n'
              << "rejection_preserved_state: " << rejection_preserved_state << '\n'
              << "invalid_token_rejected: " << invalid_token_rejected << '\n'
              << "rollback_preserved_state: " << rollback_preserved_state << '\n'
              << "states_committed: " << states_committed << '\n'
              << "paged_kv_match: " << paged_kv_match << '\n'
              << "batch_logits_match: " << logits_equal << '\n'
              << "batch_tokens_match: " << batch_match << '\n'
              << "batch_golden_match: " << golden_match << '\n';
    return position_rejected && rejection_preserved_state && invalid_token_rejected &&
        rollback_preserved_state && states_committed && paged_kv_match &&
        batch_match && golden_match ? 0 : 1;
}
