// Chapter 08：验证 overlap 预计算计划的等价、失效和同步回退。
// 职责：固定状态版本契约，不声称已经实现设备执行重叠。
// 验证入口：chapter08_overlap_scheduler_test。
#include "chapter08_scheduling/overlap_scheduler.h"

#include "chapter06_generation/request.h"
#include "chapter06_generation/runtime.h"
#include "chapter06_generation/sampler.h"
#include "cpp_llm_engine/testing/test.h"
#include "support/test_executors.h"

#include <memory>
#include <vector>

int main() {
    using namespace chapter06_generation;
    using namespace chapter08_scheduling;
    cpp_llm_engine::testing::TestSuite suite("chapter08_overlap_scheduler");

    suite.run("validated prepared steps match synchronous scheduling", [&] {
        GenerationConfig config;
        config.max_new_tokens = 2;
        Runtime sync_runtime(config);
        Runtime overlap_runtime(config);
        GreedySampler sampler;
        Request sync_request({1, 2});
        Request overlap_request({1, 2});
        auto sync_executor = std::make_shared<testing::ChunkedTokenExecutor>(5);
        auto overlap_executor = std::make_shared<testing::ChunkedTokenExecutor>(5);
        const std::vector<ScheduledRequest> sync_requests{
            {10, &sync_request, &sampler, sync_executor},
        };
        const std::vector<ScheduledRequest> overlap_requests{
            {10, &overlap_request, &sampler, overlap_executor},
        };
        OverlapScheduler overlap;

        while (!sync_request.terminal()) {
            ContinuousBatchScheduler{}.run_step(sync_runtime, sync_requests, 2);
            const auto prepared = overlap.prepare(overlap_requests, 2);
            const auto result = overlap.dispatch(
                overlap_runtime, overlap_requests, prepared);
            CPP_LLM_CHECK(suite, result.status == OverlapDispatchStatus::Executed);
        }
        CPP_LLM_CHECK(suite, overlap_request.generated() == sync_request.generated());
        CPP_LLM_CHECK(suite, overlap_request.token_history() == sync_request.token_history());
    });

    suite.run("cancellation invalidates a prepared batch", [&] {
        GenerationConfig config;
        config.max_new_tokens = 1;
        Runtime runtime(config);
        GreedySampler sampler;
        Request request({1, 2});
        auto executor = std::make_shared<testing::FixedTokenExecutor>(4);
        const std::vector<ScheduledRequest> requests{{20, &request, &sampler, executor}};
        OverlapScheduler overlap;
        const auto prepared = overlap.prepare(requests, 2);
        request.cancel();

        const auto result = overlap.dispatch(runtime, requests, prepared);
        CPP_LLM_CHECK(suite, result.status == OverlapDispatchStatus::Stale);
        CPP_LLM_CHECK(suite, executor->prefill_calls == 0);
    });

    suite.run("an earlier step invalidates another speculative plan", [&] {
        GenerationConfig config;
        config.max_new_tokens = 2;
        Runtime runtime(config);
        GreedySampler sampler;
        Request request({1, 2});
        auto executor = std::make_shared<testing::FixedTokenExecutor>(4);
        const std::vector<ScheduledRequest> requests{{30, &request, &sampler, executor}};
        OverlapScheduler overlap;
        const auto first = overlap.prepare(requests, 2);
        const auto speculative_next = overlap.prepare(requests, 1);

        CPP_LLM_CHECK(
            suite,
            overlap.dispatch(runtime, requests, first).status ==
                OverlapDispatchStatus::Executed);
        CPP_LLM_CHECK(
            suite,
            overlap.dispatch(runtime, requests, speculative_next).status ==
                OverlapDispatchStatus::Stale);
    });

    suite.run("disabled overlap uses synchronous fallback", [&] {
        GenerationConfig config;
        config.max_new_tokens = 1;
        Runtime runtime(config);
        GreedySampler sampler;
        Request request({1});
        auto executor = std::make_shared<testing::FixedTokenExecutor>(3);
        const std::vector<ScheduledRequest> requests{{40, &request, &sampler, executor}};
        OverlapScheduler overlap(false);
        const auto prepared = overlap.prepare(requests, 1);

        const auto result = overlap.dispatch(runtime, requests, prepared);
        CPP_LLM_CHECK(
            suite, result.status == OverlapDispatchStatus::SynchronousFallback);
        CPP_LLM_CHECK(suite, request.finished());
    });

    suite.run("priority aging invalidates a prepared batch", [&] {
        GenerationConfig config;
        config.max_new_tokens = 1;
        Runtime runtime(config);
        GreedySampler sampler;
        Request request({1});
        auto executor = std::make_shared<testing::FixedTokenExecutor>(3);
        const std::vector<ScheduledRequest> requests{
            {50, &request, &sampler, executor, {0, 0}},
        };
        PriorityAgingPolicy policy;
        OverlapScheduler overlap;
        const auto prepared = overlap.prepare(requests, 1, 0, nullptr, 0, &policy);
        policy.finish_step(requests, {});

        const auto result = overlap.dispatch(
            runtime, requests, prepared, {}, nullptr, &policy);
        CPP_LLM_CHECK(suite, result.status == OverlapDispatchStatus::Stale);
        CPP_LLM_CHECK(suite, executor->prefill_calls == 0);
    });

    return suite.finish();
}
