// Chapter 06：Generation Core 的命名端到端测试。
// 职责：验证 Prefill/Decode、上下文准入、EOS 和 Cache 增长。
// 验证入口：带 unit、generation 标签的 Chapter06 CTest。
#include "chapter06_generation/model_executor.h"
#include "chapter06_generation/runtime.h"
#include "chapter06_generation/sampler.h"
#include "chapter06_generation/transformer.h"

#include "cpp_llm_engine/testing/test.h"

#include <string>
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

int main() {
    using namespace chapter06_generation;
    cpp_llm_engine::testing::TestSuite suite("chapter06_generation_core");

    suite.run("assembled transformer prefills and decodes to configured length", [&] {
        GenerationConfig config;
        config.max_new_tokens = 3;
        config.use_kv_cache = true;
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
                }) == 1);
        CPP_LLM_CHECK(suite, request.generated() == TokenSequence({7}));
        CPP_LLM_CHECK(suite, streamed_tokens == TokenSequence({7}));
        CPP_LLM_CHECK(suite, streamed_steps == std::vector<std::size_t>({0}));
        CPP_LLM_CHECK(suite, observed_output_sizes == std::vector<std::size_t>({1}));
        CPP_LLM_CHECK(suite, request.history().size() == 3);
        CPP_LLM_CHECK(suite, request.finished());
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
