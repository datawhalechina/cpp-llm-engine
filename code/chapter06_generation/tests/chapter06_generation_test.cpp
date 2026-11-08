// Chapter 06：Generation Core 回归测试
// 验证 Prefill/Decode、上下文准入、EOS、取消和空 logits 失败路径。
// 入口：带 unit、generation 标签的 Chapter06 CTest。
#include "chapter06_generation/model_executor.h"
#include "chapter06_generation/runtime.h"
#include "chapter06_generation/sampler.h"
#include "chapter06_generation/transformer.h"

#include "cpp_llm_engine/testing/test.h"

#include <string>
#include <type_traits>
#include <vector>

namespace {

class FailingPrefillExecutor final
    : public chapter06_generation::RequestModelExecutor {
public:
    bool prefill(
        const chapter06_generation::TokenSequence&,
        chapter06_generation::Logits*,
        std::string* error) override {
        if (error != nullptr) *error = "synthetic prefill failure";
        return false;
    }

    bool decode(
        chapter06_generation::TokenId,
        chapter06_generation::Logits*,
        std::string*) override {
        return false;
    }
};

class FixedTokenExecutor final : public chapter06_generation::RequestModelExecutor {
public:
    bool prefill(
        const chapter06_generation::TokenSequence&,
        chapter06_generation::Logits* logits,
        std::string*) override {
        logits->assign(8, 0.0f);
        (*logits)[3] = 1.0f;
        return true;
    }

    bool decode(
        chapter06_generation::TokenId,
        chapter06_generation::Logits*,
        std::string*) override {
        ++decode_calls;
        return false;
    }

    std::size_t decode_calls = 0;
};

}  // namespace

static_assert(
    !std::is_constructible_v<
        chapter06_generation::Runtime,
        chapter06_generation::GenerationConfig,
        chapter06_generation::Transformer&&>,
    "Runtime must reject temporary Transformer objects");

int main() {
    using namespace chapter06_generation;
    cpp_llm_engine::testing::TestSuite suite("chapter06_generation_core");

    suite.run("assembled transformer prefills and decodes to configured length", [&] {
        GenerationConfig config;
        config.max_new_tokens = 3;
        Embedding embedding(32);
        PositionEncoding position_encoding(16);
        Transformer transformer(2, embedding, position_encoding);
        Runtime runtime(config, transformer);
        Request request({1, 2, 3});
        GreedySampler sampler;

        CPP_LLM_CHECK(suite, embedding.ready());
        CPP_LLM_CHECK(suite, position_encoding.ready());
        CPP_LLM_CHECK(suite, transformer.ready());
        CPP_LLM_CHECK(suite, transformer.layer_count() == 2);
        CPP_LLM_CHECK(suite, transformer.embedding().vocab_size() == 32);
        CPP_LLM_CHECK(suite, transformer.position_encoding().max_length() == 16);
        CPP_LLM_CHECK(suite, runtime.prefill(request) == 3);
        CPP_LLM_CHECK(suite, request.prefilled());
        CPP_LLM_CHECK(suite, request.phase() == RequestPhase::Prefilled);
        CPP_LLM_CHECK(suite, request.history().size() == 3);
        CPP_LLM_CHECK(suite, runtime.prefill(request) == 0);
        CPP_LLM_CHECK(suite, request.history().size() == 3);
        CPP_LLM_CHECK(suite, runtime.decode(request, sampler) == 3);
        CPP_LLM_CHECK(suite, request.generated_size() == 3);
        CPP_LLM_CHECK(suite, request.phase() == RequestPhase::Finished);
        CPP_LLM_CHECK(suite, request.history().size() == 6);
        CPP_LLM_CHECK(suite, request.finished());
    });

    suite.run("context admission rejects prompt plus generation overflow", [&] {
        Embedding embedding(32);
        PositionEncoding position_encoding(16);
        Transformer transformer(2, embedding, position_encoding);
        GenerationConfig config;
        config.max_new_tokens = 2;
        config.max_context_tokens = 4;
        Runtime runtime(config, transformer);
        Request request({1, 2, 3});

        CPP_LLM_CHECK(suite, runtime.prefill(request) == 0);
        CPP_LLM_CHECK(suite, request.phase() == RequestPhase::Rejected);
        CPP_LLM_CHECK(suite, request.terminal());
        CPP_LLM_CHECK(suite, request.history().empty());
    });

    suite.run("provider EOS finishes generation after one token", [&] {
        GenerationConfig config;
        config.max_new_tokens = 4;
        config.eos_token_id = 7;
        Runtime runtime(config);
        Request request({4, 5});
        GreedySampler sampler;
        const auto provider = [](const Request&, std::size_t) {
            Logits logits(8, 0.0f);
            logits[7] = 1.0f;
            return logits;
        };
        std::vector<TokenId> streamed_tokens;
        std::vector<std::size_t> streamed_steps;
        std::vector<std::size_t> observed_output_sizes;
        std::vector<std::size_t> observed_history_sizes;
        std::vector<RequestPhase> observed_phases;

        CPP_LLM_CHECK(
            suite,
            runtime.generate(
                request,
                sampler,
                provider,
                [&](TokenId token, std::size_t step) {
                    streamed_tokens.push_back(token);
                    streamed_steps.push_back(step);
                    observed_output_sizes.push_back(request.generated_size());
                    observed_history_sizes.push_back(request.history().size());
                    observed_phases.push_back(request.phase());
                }) == 1);
        CPP_LLM_CHECK(suite, request.generated() == TokenSequence({7}));
        CPP_LLM_CHECK(suite, streamed_tokens == TokenSequence({7}));
        CPP_LLM_CHECK(suite, streamed_steps == std::vector<std::size_t>({0}));
        CPP_LLM_CHECK(suite, observed_output_sizes == std::vector<std::size_t>({1}));
        CPP_LLM_CHECK(suite, observed_history_sizes == std::vector<std::size_t>({3}));
        CPP_LLM_CHECK(suite, observed_phases == std::vector<RequestPhase>({RequestPhase::Finished}));
        CPP_LLM_CHECK(suite, request.history().size() == 3);
        CPP_LLM_CHECK(suite, request.finished());
    });

    suite.run("empty provider logits fail without generated output", [&] {
        GenerationConfig config;
        config.max_new_tokens = 4;
        Runtime runtime(config);
        GreedySampler sampler;
        const LogitsProvider empty_provider = [](const Request&, std::size_t) {
            return Logits{};
        };
        Request request({1});
        CPP_LLM_CHECK(
            suite,
            runtime.generate(request, sampler, empty_provider) == 0);
        CPP_LLM_CHECK(suite, request.failed());
        CPP_LLM_CHECK(suite, request.generated().empty());
    });

    suite.run("callback cancellation stops after the committed token", [&] {
        GenerationConfig config;
        config.max_new_tokens = 4;
        Runtime runtime(config);
        GreedySampler sampler;
        const LogitsProvider token_provider = [](const Request&, std::size_t) {
            return Logits{0.0F, 1.0F};
        };
        Request request({1});
        std::size_t callback_count = 0;
        CPP_LLM_CHECK(
            suite,
            runtime.generate(
                request,
                sampler,
                token_provider,
                [&](TokenId, std::size_t) {
                    ++callback_count;
                    request.cancel();
                }) == 1);
        CPP_LLM_CHECK(suite, request.cancelled());
        CPP_LLM_CHECK(suite, request.generated() == TokenSequence({1}));
        CPP_LLM_CHECK(suite, request.history().tokens() == TokenSequence({1, 1}));
        CPP_LLM_CHECK(suite, callback_count == 1);
    });

    suite.run("decode_one commits one sampled token", [&] {
        GenerationConfig config;
        config.max_new_tokens = 4;
        Runtime runtime(config);
        GreedySampler sampler;
        const LogitsProvider token_provider = [](const Request&, std::size_t) {
            return Logits{0.0F, 1.0F};
        };
        Request request({1});
        request.mark_prefilled();
        CPP_LLM_CHECK(
            suite,
            runtime.decode_one(request, sampler, token_provider) == 1);
        CPP_LLM_CHECK(suite, request.generated() == TokenSequence({1}));
    });

    suite.run("batch logits validate every row before committing tokens", [&] {
        GenerationConfig config;
        config.max_new_tokens = 2;
        Runtime runtime(config);
        GreedySampler sampler;
        Request request_a({1});
        Request request_b({2});
        request_a.mark_prefilled();
        request_b.mark_prefilled();
        const std::vector<BatchDecodeItem> items{
            {&request_a, &sampler}, {&request_b, &sampler}};

        const auto committed = runtime.commit_batch_logits(
            items, LogitsBatch{{0.0F, 2.0F}, {3.0F, 0.0F}});
        CPP_LLM_CHECK(suite, committed.tokens.size() == 2);
        CPP_LLM_CHECK(suite, request_a.generated() == TokenSequence({1}));
        CPP_LLM_CHECK(suite, request_b.generated() == TokenSequence({0}));

        Request invalid_a({1});
        Request invalid_b({2});
        invalid_a.mark_prefilled();
        invalid_b.mark_prefilled();
        const std::vector<BatchDecodeItem> invalid_items{
            {&invalid_a, &sampler}, {&invalid_b, &sampler}};
        const auto rejected = runtime.commit_batch_logits(
            invalid_items, LogitsBatch{{0.0F, 1.0F}, {}});
        CPP_LLM_CHECK(suite, rejected.tokens.empty());
        CPP_LLM_CHECK(suite, invalid_a.generated().empty());
        CPP_LLM_CHECK(suite, invalid_b.generated().empty());

        const std::vector<BatchDecodeItem> duplicate_items{
            {&invalid_a, &sampler}, {&invalid_a, &sampler}};
        CPP_LLM_CHECK(
            suite,
            runtime.commit_batch_logits(
                duplicate_items, LogitsBatch{{1.0F}, {1.0F}}).tokens.empty());
        CPP_LLM_CHECK(suite, invalid_a.generated().empty());
    });

    suite.run("model prefill failure terminates request without output", [&] {
        GenerationConfig config;
        config.max_new_tokens = 2;
        Runtime runtime(config);
        Request request({1});
        GreedySampler sampler;
        FailingPrefillExecutor executor;

        CPP_LLM_CHECK(suite, runtime.generate(request, sampler, executor) == 0);
        CPP_LLM_CHECK(suite, request.failed());
        CPP_LLM_CHECK(suite, request.failure_reason() == "synthetic prefill failure");
        CPP_LLM_CHECK(suite, request.generated().empty());
    });

    suite.run("model decode requires a previously generated token", [&] {
        GenerationConfig config;
        config.max_new_tokens = 2;
        Runtime runtime(config);
        Request request({1});
        request.mark_prefilled();
        GreedySampler sampler;
        FixedTokenExecutor executor;

        CPP_LLM_CHECK(suite, runtime.decode_one(request, sampler, executor) == 0);
        CPP_LLM_CHECK(suite, request.failed());
        CPP_LLM_CHECK(
            suite,
            request.failure_reason() ==
                "model decode requires a previously generated token");
        CPP_LLM_CHECK(suite, executor.decode_calls == 0);
    });

    return suite.finish();
}
