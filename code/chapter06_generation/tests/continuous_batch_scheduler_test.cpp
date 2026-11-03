// Chapter 06：Continuous Batch Planner 和动态推进组件测试。
// 职责：验证预算选择、Prefill/Decode 混排和终态停止。
// 验证入口：带 component、scheduler、batch 标签的 Chapter06 CTest。
#include "chapter06_generation/serving_preview.h"

#include "chapter06_generation/request.h"
#include "chapter06_generation/runtime.h"
#include "chapter06_generation/sampler.h"
#include "cpp_llm_engine/testing/test.h"
#include "support/serving_test_executors.h"

#include <memory>
#include <stdexcept>
#include <vector>

int main() {
    using namespace chapter06_generation;
    using testing::FixedTokenExecutor;
    cpp_llm_engine::testing::TestSuite suite("chapter06_continuous_batch_scheduler");

    suite.run("planner respects token budget and skips terminal request", [&] {
        Request request_a({1, 2});
        Request request_b({3, 4, 5});
        Request cancelled({4});
        cancelled.cancel();
        GreedySampler sampler;
        const std::vector<ScheduledRequest> requests = {
            {7, &request_a, &sampler, std::make_shared<FixedTokenExecutor>(5)},
            {8, &request_b, &sampler, std::make_shared<FixedTokenExecutor>(6)},
            {9, &cancelled, &sampler, {}},
        };

        const auto plan = StepBatchPlanner{}.build(requests, 2);
        CPP_LLM_REQUIRE(suite, plan.candidates.size() == 1);
        CPP_LLM_CHECK(suite, plan.scheduled_tokens == 2);
        CPP_LLM_CHECK(suite, plan.deferred_requests == 1);
        CPP_LLM_CHECK(suite, plan.candidates[0].request_id == 7);
        CPP_LLM_CHECK(suite, plan.candidates[0].prefill);
    });

    suite.run("successive steps mix decode and prefill until completion", [&] {
        GenerationConfig config;
        config.max_new_tokens = 2;
        config.max_context_tokens = 16;
        Runtime runtime(config);
        GreedySampler sampler;
        Request request_a({1, 2});
        Request request_b({3, 4, 5});
        Request cancelled({4});
        cancelled.cancel();
        Request request_c({4, 5});
        auto executor_a = std::make_shared<FixedTokenExecutor>(5);
        auto executor_b = std::make_shared<FixedTokenExecutor>(6);
        auto executor_c = std::make_shared<FixedTokenExecutor>(7);
        std::vector<ScheduledRequest> requests = {
            {7, &request_a, &sampler, executor_a},
            {8, &request_b, &sampler, executor_b},
            {9, &cancelled, &sampler, {}},
        };
        const ContinuousBatchScheduler scheduler;

        const auto first = scheduler.run_step(runtime, requests, 2);
        CPP_LLM_CHECK(suite, first.prefill_requests == 1);
        CPP_LLM_CHECK(suite, first.generated_tokens == 1);
        CPP_LLM_CHECK(suite, first.deferred_requests == 1);
        CPP_LLM_CHECK(suite, request_a.generated() == TokenSequence({5}));

        requests.push_back({10, &request_c, &sampler, executor_c});

        const auto mixed_plan = StepBatchPlanner{}.build(requests, 4);
        CPP_LLM_REQUIRE(suite, mixed_plan.candidates.size() == 2);
        CPP_LLM_CHECK(suite, mixed_plan.decode_requests == 1);
        CPP_LLM_CHECK(suite, mixed_plan.prefill_requests == 1);
        CPP_LLM_CHECK(suite, mixed_plan.candidates[0].request_id == 7);
        CPP_LLM_CHECK(suite, !mixed_plan.candidates[0].prefill);
        CPP_LLM_CHECK(suite, mixed_plan.candidates[1].request_id == 8);
        CPP_LLM_CHECK(suite, mixed_plan.candidates[1].prefill);

        const auto mixed = scheduler.run_step(runtime, requests, 4);
        CPP_LLM_CHECK(suite, mixed.generated_tokens == 2);
        CPP_LLM_CHECK(suite, request_a.finished());
        CPP_LLM_CHECK(suite, request_b.generated() == TokenSequence({6}));
        CPP_LLM_CHECK(suite, !request_c.prefilled());

        const auto dynamic = scheduler.run_step(runtime, requests, 3);
        CPP_LLM_CHECK(suite, dynamic.decode_requests == 1);
        CPP_LLM_CHECK(suite, dynamic.prefill_requests == 1);
        CPP_LLM_CHECK(suite, dynamic.generated_tokens == 2);
        CPP_LLM_CHECK(suite, request_b.finished());
        CPP_LLM_CHECK(suite, request_c.generated() == TokenSequence({7}));
        CPP_LLM_CHECK(suite, scheduler.run_step(runtime, requests, 1).generated_tokens == 1);
        CPP_LLM_CHECK(suite, request_c.finished());
        CPP_LLM_CHECK(suite, executor_a->prefill_calls == 1 && executor_a->decode_calls == 1);
        CPP_LLM_CHECK(suite, executor_b->prefill_calls == 1 && executor_b->decode_calls == 1);
        CPP_LLM_CHECK(suite, executor_c->prefill_calls == 1 && executor_c->decode_calls == 1);
        CPP_LLM_CHECK(suite, scheduler.run_step(runtime, requests, 4).scheduled_requests == 0);
    });

    suite.run("zero token budget is rejected", [&] {
        bool rejected = false;
        try {
            (void)StepBatchPlanner{}.build({}, 0);
        } catch (const std::invalid_argument&) {
            rejected = true;
        }
        CPP_LLM_CHECK(suite, rejected);
    });

    return suite.finish();
}
