// Chapter 06：RoundRobin Scheduler 命名组件测试。
// 职责：验证轮转顺序、终态跳过、执行器隔离和模型失败传播。
// 验证入口：Chapter06 CTest 或带 component、scheduler 标签的 CTest。
#include "chapter06_generation/model_executor.h"
#include "chapter06_generation/runtime.h"
#include "chapter06_generation/sampler.h"
#include "chapter06_generation/scheduler.h"
#include "chapter06_generation/transformer.h"

#include "cpp_llm_engine/testing/test.h"

#include <memory>
#include <string>
#include <vector>

namespace {

using cpp_llm_engine::testing::TestSuite;

class FixedExecutor final : public chapter06_generation::RequestModelExecutor {
public:
    explicit FixedExecutor(chapter06_generation::TokenId token) : token_(token) {}

    bool prefill(const chapter06_generation::TokenSequence&,
                 chapter06_generation::Logits* logits,
                 std::string*) override {
        ++prefill_calls;
        *logits = make_logits();
        return true;
    }

    bool decode(chapter06_generation::TokenId,
                chapter06_generation::Logits* logits,
                std::string*) override {
        ++decode_calls;
        *logits = make_logits();
        return true;
    }

    std::size_t prefill_calls = 0;
    std::size_t decode_calls = 0;

private:
    chapter06_generation::Logits make_logits() const {
        chapter06_generation::Logits logits(8, 0.0f);
        logits[static_cast<std::size_t>(token_)] = 1.0f;
        return logits;
    }

    chapter06_generation::TokenId token_;
};

class FailingExecutor final : public chapter06_generation::RequestModelExecutor {
public:
    bool prefill(const chapter06_generation::TokenSequence&,
                 chapter06_generation::Logits*,
                 std::string* error) override {
        *error = "synthetic prefill failure";
        return false;
    }

    bool decode(chapter06_generation::TokenId,
                chapter06_generation::Logits*,
                std::string*) override {
        return false;
    }
};

chapter06_generation::GenerationConfig test_config() {
    chapter06_generation::GenerationConfig config;
    config.max_new_tokens = 2;
    config.use_kv_cache = true;
    return config;
}

}  // namespace

int main() {
    using namespace chapter06_generation;
    TestSuite suite("chapter06_round_robin_scheduler");

    suite.run("round robin alternates active requests and skips terminal request", [&] {
        Embedding embedding(32);
        PositionEncoding position_encoding(16);
        Transformer transformer(2, embedding, position_encoding);
        Runtime runtime(test_config(), transformer);
        GreedySampler sampler;
        Request request_a({1, 2, 3});
        Request request_b({4, 5});
        Request cancelled({6});
        cancelled.cancel();
        const std::vector<ScheduledRequest> requests = {
            {0, &request_a, &sampler, {}},
            {1, &request_b, &sampler, {}},
            {2, &cancelled, &sampler, {}},
        };

        std::vector<std::size_t> request_ids;
        std::vector<std::size_t> steps;
        std::vector<TokenId> tokens;
        const RoundRobinScheduler scheduler;
        const auto stats = scheduler.run_with_stats(
            runtime,
            requests,
            [&](std::size_t request_id, TokenId token, std::size_t step) {
                request_ids.push_back(request_id);
                tokens.push_back(token);
                steps.push_back(step);
            });

        CPP_LLM_CHECK(suite, stats.scheduled_requests == 3);
        CPP_LLM_CHECK(suite, stats.prefetched_requests == 2);
        CPP_LLM_CHECK(suite, stats.generated_tokens == 4);
        CPP_LLM_CHECK(suite, stats.completed_requests == 2);
        CPP_LLM_CHECK(suite, request_a.finished());
        CPP_LLM_CHECK(suite, request_b.finished());
        CPP_LLM_CHECK(suite, request_a.generated_size() == 2);
        CPP_LLM_CHECK(suite, request_b.generated_size() == 2);
        CPP_LLM_CHECK(suite, request_a.cache().size() == 5);
        CPP_LLM_CHECK(suite, request_b.cache().size() == 4);
        CPP_LLM_CHECK(suite, cancelled.cache().empty());
        CPP_LLM_CHECK(suite, request_ids == std::vector<std::size_t>({0, 1, 0, 1}));
        CPP_LLM_CHECK(suite, steps == std::vector<std::size_t>({0, 0, 1, 1}));
        CPP_LLM_CHECK(suite, tokens == std::vector<TokenId>({2, 2, 2, 2}));
    });

    suite.run("each request uses its own model executor", [&] {
        Runtime runtime(test_config());
        GreedySampler sampler;
        Request request_a({1, 2});
        Request request_b({3, 4});
        auto executor_a = std::make_shared<FixedExecutor>(3);
        auto executor_b = std::make_shared<FixedExecutor>(4);
        const RoundRobinScheduler scheduler;
        const auto stats = scheduler.run_with_stats(
            runtime,
            {
                {10, &request_a, &sampler, executor_a},
                {11, &request_b, &sampler, executor_b},
            });

        CPP_LLM_CHECK(suite, stats.prefetched_requests == 2);
        CPP_LLM_CHECK(suite, stats.generated_tokens == 4);
        CPP_LLM_CHECK(suite, stats.completed_requests == 2);
        CPP_LLM_CHECK(suite, request_a.generated() == TokenSequence({3, 3}));
        CPP_LLM_CHECK(suite, request_b.generated() == TokenSequence({4, 4}));
        CPP_LLM_CHECK(suite, executor_a->prefill_calls == 1);
        CPP_LLM_CHECK(suite, executor_a->decode_calls == 1);
        CPP_LLM_CHECK(suite, executor_b->prefill_calls == 1);
        CPP_LLM_CHECK(suite, executor_b->decode_calls == 1);
    });

    suite.run("prefill failure terminates request without output", [&] {
        Runtime runtime(test_config());
        GreedySampler sampler;
        Request request({1});
        FailingExecutor executor;

        CPP_LLM_CHECK(suite, runtime.generate(request, sampler, executor) == 0);
        CPP_LLM_CHECK(suite, request.failed());
        CPP_LLM_CHECK(suite, request.failure_reason() == "synthetic prefill failure");
        CPP_LLM_CHECK(suite, request.generated().empty());
    });

    suite.run("decode requires a previously generated token", [&] {
        Runtime runtime(test_config());
        GreedySampler sampler;
        Request request({1});
        request.mark_prefilled();
        FixedExecutor executor(3);

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
