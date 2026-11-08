// Chapter 08：验证预算计划、混合执行与 Chunked Prefill 公平性。
// 职责：固定 StepBatchPlan 和逐步推进结果，不测试网络 admission。
// 验证入口：chapter08_step_scheduler_test。
#include "chapter08_scheduling/step_scheduler.h"

#include "chapter06_generation/request.h"
#include "chapter06_generation/runtime.h"
#include "chapter06_generation/sampler.h"
#include "cpp_llm_engine/testing/test.h"
#include "support/test_executors.h"

#include <memory>
#include <stdexcept>
#include <vector>

int main() {
    using namespace chapter06_generation;
    using namespace chapter08_scheduling;
    cpp_llm_engine::testing::TestSuite suite("chapter08_step_scheduler");

    suite.run("planner respects token budget without mutating requests", [&] {
        Request request_a({1, 2});
        Request request_b({3, 4, 5});
        GreedySampler sampler;
        const std::vector<ScheduledRequest> requests{
            {7, &request_a, &sampler, std::make_shared<testing::FixedTokenExecutor>(5)},
            {8, &request_b, &sampler, std::make_shared<testing::FixedTokenExecutor>(6)},
        };
        const auto plan = StepBatchPlanner{}.build(requests, 2);
        CPP_LLM_REQUIRE(suite, plan.candidates.size() == 1);
        CPP_LLM_CHECK(suite, plan.candidates.front().request_id == 7);
        CPP_LLM_CHECK(suite, plan.scheduled_tokens == 2);
        CPP_LLM_CHECK(suite, plan.deferred_requests == 1);
        CPP_LLM_CHECK(suite, request_a.prefill_offset() == 0);
        CPP_LLM_CHECK(suite, request_a.generated().empty());
    });

    suite.run("cached prefix breaks equal priority prefill ties", [&] {
        Request cold({1, 2});
        Request cached({1, 2});
        GreedySampler sampler;
        const std::vector<ScheduledRequest> requests{
            {10,
             &cold,
             &sampler,
             std::make_shared<testing::FixedTokenExecutor>(5),
             {0, 0, 0}},
            {11,
             &cached,
             &sampler,
             std::make_shared<testing::FixedTokenExecutor>(6),
             {0, 1, 2}},
        };

        const auto plan = StepBatchPlanner{}.build(requests, 2);
        CPP_LLM_REQUIRE(suite, plan.candidates.size() == 1);
        CPP_LLM_CHECK(suite, plan.candidates.front().request_id == 11);
        CPP_LLM_CHECK(suite, plan.candidates.front().token_count == 2);
        CPP_LLM_CHECK(suite, cold.prefill_offset() == 0);
        CPP_LLM_CHECK(suite, cached.prefill_offset() == 0);
    });

    suite.run("successive steps mix decode and prefill", [&] {
        GenerationConfig config;
        config.max_new_tokens = 2;
        config.max_context_tokens = 16;
        Runtime runtime(config);
        GreedySampler sampler;
        Request request_a({1, 2});
        Request request_b({3, 4, 5});
        auto executor_a = std::make_shared<testing::FixedTokenExecutor>(5);
        auto executor_b = std::make_shared<testing::FixedTokenExecutor>(6);
        const std::vector<ScheduledRequest> requests{
            {7, &request_a, &sampler, executor_a},
            {8, &request_b, &sampler, executor_b},
        };
        const ContinuousBatchScheduler scheduler;
        const auto first = scheduler.run_step(runtime, requests, 2);
        CPP_LLM_CHECK(suite, first.prefill_requests == 1);
        CPP_LLM_CHECK(suite, first.generated_tokens == 1);

        const auto mixed = scheduler.run_step(runtime, requests, 4);
        CPP_LLM_CHECK(suite, mixed.decode_requests == 1);
        CPP_LLM_CHECK(suite, mixed.prefill_requests == 1);
        CPP_LLM_CHECK(suite, request_a.finished());
        CPP_LLM_CHECK(suite, request_b.generated() == TokenSequence({6}));
    });

    suite.run("one token budget reserves a chunked prefill opportunity", [&] {
        GenerationConfig config;
        config.max_new_tokens = 3;
        Runtime runtime(config);
        GreedySampler sampler;
        Request decoding({1});
        auto decode_executor = std::make_shared<testing::FixedTokenExecutor>(3);
        CPP_LLM_CHECK(suite, runtime.decode_one(decoding, sampler, *decode_executor) == 1);
        Request waiting({2, 3, 4, 5});
        auto chunk_executor = std::make_shared<testing::ChunkedTokenExecutor>(4);

        const auto stats = ContinuousBatchScheduler{}.run_step(
            runtime,
            {{60, &decoding, &sampler, decode_executor},
             {61, &waiting, &sampler, chunk_executor}},
            1,
            1);
        CPP_LLM_CHECK(suite, stats.prefill_requests == 1);
        CPP_LLM_CHECK(suite, stats.decode_requests == 0);
        CPP_LLM_CHECK(suite, waiting.prefill_offset() == 1);
        CPP_LLM_CHECK(suite, waiting.generated().empty());
    });

    suite.run("two long prefills alternate and finish together", [&] {
        GenerationConfig config;
        config.max_new_tokens = 1;
        config.max_context_tokens = 16;
        Runtime runtime(config);
        GreedySampler sampler;
        Request request_a({1, 2, 3, 4, 5});
        Request request_b({6, 7, 1, 2, 3});
        auto executor_a = std::make_shared<testing::ChunkedTokenExecutor>(4);
        auto executor_b = std::make_shared<testing::ChunkedTokenExecutor>(5);
        const std::vector<ScheduledRequest> requests{
            {70, &request_a, &sampler, executor_a},
            {71, &request_b, &sampler, executor_b},
        };
        const ContinuousBatchScheduler scheduler;

        scheduler.run_step(runtime, requests, 2);
        CPP_LLM_CHECK(suite, request_a.prefill_offset() == 2);
        CPP_LLM_CHECK(suite, request_b.prefill_offset() == 0);
        scheduler.run_step(runtime, requests, 2);
        CPP_LLM_CHECK(suite, request_b.prefill_offset() == 2);
        scheduler.run_step(runtime, requests, 2);
        CPP_LLM_CHECK(suite, request_a.prefill_offset() == 4);
        scheduler.run_step(runtime, requests, 2);
        CPP_LLM_CHECK(suite, request_b.prefill_offset() == 4);
        const auto final_step = scheduler.run_step(runtime, requests, 2);

        CPP_LLM_CHECK(suite, final_step.prefill_requests == 2);
        CPP_LLM_CHECK(suite, final_step.generated_tokens == 2);
        CPP_LLM_CHECK(suite, request_a.finished() && request_b.finished());
        CPP_LLM_CHECK(suite, request_a.generated() == TokenSequence({4}));
        CPP_LLM_CHECK(suite, request_b.generated() == TokenSequence({5}));
        CPP_LLM_CHECK(
            suite, executor_a->chunk_sizes == std::vector<std::size_t>({2, 2, 1}));
        CPP_LLM_CHECK(
            suite, executor_b->chunk_sizes == std::vector<std::size_t>({2, 2, 1}));
    });

    suite.run("zero token budget is rejected", [&] {
        bool rejected = false;
        try {
            static_cast<void>(StepBatchPlanner{}.build({}, 0));
        } catch (const std::invalid_argument&) {
            rejected = true;
        }
        CPP_LLM_CHECK(suite, rejected);
    });

    suite.run("prefill progress cannot exceed prompt length", [&] {
        bool rejected = false;
        try {
            Request request({1});
            request.advance_prefill(2);
        } catch (const std::out_of_range&) {
            rejected = true;
        }
        CPP_LLM_CHECK(suite, rejected);
    });

    suite.run("block pressure defers request until capacity is released", [&] {
        GenerationConfig config;
        config.max_new_tokens = 1;
        config.max_context_tokens = 16;
        Runtime runtime(config);
        GreedySampler sampler;
        Request request_a({1, 2, 3});
        Request request_b({4, 5});
        auto executor_a = std::make_shared<testing::FixedTokenExecutor>(5);
        auto executor_b = std::make_shared<testing::FixedTokenExecutor>(6);
        const std::vector<ScheduledRequest> requests{
            {30, &request_a, &sampler, executor_a},
            {31, &request_b, &sampler, executor_b},
        };
        chapter07_kv_cache::KvBlockPool pool(2, 2);
        const ContinuousBatchScheduler scheduler;

        const auto limited = scheduler.run_step(runtime, requests, 5, 0, {}, &pool);
        CPP_LLM_CHECK(suite, limited.scheduled_requests == 1);
        CPP_LLM_CHECK(suite, limited.deferred_requests == 1);
        CPP_LLM_CHECK(suite, request_a.finished());
        CPP_LLM_CHECK(suite, !request_b.prefilled());
        CPP_LLM_CHECK(suite, pool.release(30));

        const auto admitted = scheduler.run_step(runtime, requests, 2, 0, {}, &pool);
        CPP_LLM_CHECK(suite, admitted.scheduled_requests == 1);
        CPP_LLM_CHECK(suite, request_b.finished());
        CPP_LLM_CHECK(suite, pool.locate(31, 0).block == 0);
    });

    suite.run("chunked prefill keeps request model and block table consistent", [&] {
        GenerationConfig config;
        config.max_new_tokens = 1;
        config.max_context_tokens = 16;
        Runtime runtime(config);
        GreedySampler sampler;
        Request request_a({1, 2, 3, 4, 5});
        Request request_b({6, 7, 1, 2, 3});
        auto executor_a = std::make_shared<testing::ChunkedTokenExecutor>(4);
        auto executor_b = std::make_shared<testing::ChunkedTokenExecutor>(5);
        const std::vector<ScheduledRequest> requests{
            {40, &request_a, &sampler, executor_a},
            {41, &request_b, &sampler, executor_b},
        };
        chapter07_kv_cache::KvBlockPool pool(6, 2);
        const ContinuousBatchScheduler scheduler;

        const auto first = scheduler.run_step(runtime, requests, 2, 0, {}, &pool);
        CPP_LLM_CHECK(suite, first.scheduled_tokens == 2);
        const auto partial = inspect_request_execution_state(
            40, request_a, executor_a.get(), &pool);
        CPP_LLM_CHECK(suite, partial.consistent());
        CPP_LLM_CHECK(suite, partial.token_history_tokens == 2);
        CPP_LLM_CHECK(suite, partial.model.committed_tokens == 2);
        CPP_LLM_CHECK(suite, partial.block_table_tokens == 2);

        for (int step = 0; step < 4; ++step) {
            scheduler.run_step(runtime, requests, 2, 0, {}, &pool);
        }
        CPP_LLM_CHECK(suite, request_a.finished() && request_b.finished());
        CPP_LLM_CHECK(suite, pool.token_count(40) == 5);
        CPP_LLM_CHECK(suite, pool.token_count(41) == 5);
        CPP_LLM_CHECK(
            suite,
            inspect_request_execution_state(40, request_a, executor_a.get(), &pool)
                .consistent());
    });

    return suite.finish();
}
