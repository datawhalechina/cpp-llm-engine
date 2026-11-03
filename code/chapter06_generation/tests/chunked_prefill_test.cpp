// Chapter 06：Chunked Prefill 推进和三层状态一致性测试。
// 职责：验证请求历史、模型已提交 token 和 BlockTable 同步推进。
// 验证入口：带 component、prefill、cache 标签的 Chapter06 CTest。
#include "chapter06_generation/serving_preview.h"

#include "chapter06_generation/request.h"
#include "chapter06_generation/runtime.h"
#include "chapter06_generation/sampler.h"
#include "cpp_llm_engine/testing/test.h"
#include "support/serving_test_executors.h"

#include <memory>
#include <vector>

int main() {
    using namespace chapter06_generation;
    using testing::ChunkedTokenExecutor;
    cpp_llm_engine::testing::TestSuite suite("chapter06_chunked_prefill");

    suite.run("two requests alternate chunks and converge to consistent state", [&] {
        GenerationConfig config;
        config.max_new_tokens = 1;
        config.max_context_tokens = 16;
        Runtime runtime(config);
        GreedySampler sampler;
        Request request_a({1, 2, 3, 4, 5});
        Request request_b({6, 7, 1, 2, 3});
        auto executor_a = std::make_shared<ChunkedTokenExecutor>(4);
        auto executor_b = std::make_shared<ChunkedTokenExecutor>(5);
        const std::vector<ScheduledRequest> requests = {
            {40, &request_a, &sampler, executor_a},
            {41, &request_b, &sampler, executor_b},
        };
        KvBlockPool pool(6, 2);
        const ContinuousBatchScheduler scheduler;

        const auto first = scheduler.run_step(runtime, requests, 2, 0, {}, &pool);
        CPP_LLM_CHECK(suite, first.scheduled_tokens == 2);
        CPP_LLM_CHECK(suite, first.generated_tokens == 0);
        CPP_LLM_CHECK(suite, request_a.prefill_offset() == 2);
        CPP_LLM_CHECK(suite, request_b.prefill_offset() == 0);
        CPP_LLM_CHECK(suite, pool.token_count(40) == 2);
        const auto partial = inspect_request_execution_state(
            40, request_a, executor_a.get(), &pool);
        CPP_LLM_CHECK(suite, partial.token_history_tokens == 2);
        CPP_LLM_CHECK(suite, partial.model.committed_tokens == 2);
        CPP_LLM_CHECK(suite, partial.block_table_tokens == 2);
        CPP_LLM_CHECK(suite, partial.consistent());

        const auto second = scheduler.run_step(runtime, requests, 2, 0, {}, &pool);
        CPP_LLM_CHECK(suite, second.generated_tokens == 0);
        CPP_LLM_CHECK(suite, request_a.prefill_offset() == 2);
        CPP_LLM_CHECK(suite, request_b.prefill_offset() == 2);
        CPP_LLM_CHECK(suite, pool.token_count(41) == 2);

        (void)scheduler.run_step(runtime, requests, 2, 0, {}, &pool);
        (void)scheduler.run_step(runtime, requests, 2, 0, {}, &pool);
        const auto finish = scheduler.run_step(runtime, requests, 2, 0, {}, &pool);
        CPP_LLM_CHECK(suite, finish.scheduled_tokens == 2);
        CPP_LLM_CHECK(suite, finish.generated_tokens == 2);
        CPP_LLM_CHECK(suite, request_a.finished() && request_b.finished());
        CPP_LLM_CHECK(
            suite, scheduler.run_step(runtime, requests, 2, 0, {}, &pool)
                       .scheduled_requests == 0);
        CPP_LLM_CHECK(
            suite, executor_a->chunk_sizes == std::vector<std::size_t>({2, 2, 1}));
        CPP_LLM_CHECK(
            suite, executor_a->starts == std::vector<bool>({true, false, false}));
        CPP_LLM_CHECK(
            suite, executor_a->finals == std::vector<bool>({false, false, true}));
        CPP_LLM_CHECK(
            suite, executor_b->chunk_sizes == std::vector<std::size_t>({2, 2, 1}));
        CPP_LLM_CHECK(suite, pool.token_count(40) == 5);
        CPP_LLM_CHECK(suite, pool.token_count(41) == 5);

        const auto final_state = inspect_request_execution_state(
            40, request_a, executor_a.get(), &pool);
        CPP_LLM_CHECK(suite, final_state.token_history_tokens == 6);
        CPP_LLM_CHECK(suite, final_state.model.committed_tokens == 5);
        CPP_LLM_CHECK(suite, final_state.block_table_tokens == 5);
        CPP_LLM_CHECK(suite, final_state.expected_model_tokens == 5);
        CPP_LLM_CHECK(suite, final_state.consistent());
    });

    return suite.finish();
}
