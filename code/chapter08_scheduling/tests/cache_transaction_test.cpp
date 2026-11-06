// Chapter 08：验证调度、模型状态与 Chapter07 BlockTable 的跨章事务。
// 职责：检查预留槽位、提交计数和执行失败回滚。
// 验证入口：chapter08_cache_transaction_test。
#include "chapter08_scheduling/step_scheduler.h"

#include "chapter06_generation/request.h"
#include "chapter06_generation/runtime.h"
#include "chapter06_generation/sampler.h"
#include "chapter07_kv_cache/block_table.h"
#include "cpp_llm_engine/testing/test.h"
#include "support/test_executors.h"

#include <memory>
#include <string>
#include <vector>

int main() {
    using namespace chapter06_generation;
    using namespace chapter08_scheduling;
    cpp_llm_engine::testing::TestSuite suite("chapter08_cache_transaction");

    suite.run("decode input requires a reserved write slot", [&] {
        GenerationConfig config;
        config.max_new_tokens = 2;
        config.max_context_tokens = 16;
        Runtime runtime(config);
        GreedySampler sampler;
        Request request({1, 2});
        auto executor = std::make_shared<testing::ChunkedTokenExecutor>(6);
        chapter07_kv_cache::KvBlockPool pool(2, 2);
        const std::vector<ScheduledRequest> requests{{45, &request, &sampler, executor}};
        const ContinuousBatchScheduler scheduler;

        CPP_LLM_CHECK(
            suite, scheduler.run_step(runtime, requests, 2, 0, {}, &pool).generated_tokens == 1);
        const auto after_prefill = inspect_request_execution_state(
            45, request, executor.get(), &pool);
        CPP_LLM_CHECK(suite, after_prefill.consistent());
        CPP_LLM_CHECK(suite, after_prefill.model.committed_tokens == 2);

        const auto plan = StepBatchPlanner{}.build(requests, 1);
        ExecutorBatchInput input;
        std::string error;
        CPP_LLM_CHECK(
            suite, !build_decode_executor_batch_input(plan, requests, pool, &input, &error));
        CPP_LLM_CHECK(suite, error == "decode batch has no reserved KV write slot");

        CPP_LLM_CHECK(suite, pool.reserve(45, 3));
        CPP_LLM_CHECK(
            suite, build_decode_executor_batch_input(plan, requests, pool, &input, &error));
        CPP_LLM_REQUIRE(suite, input.tokens.size() == 1);
        CPP_LLM_CHECK(suite, input.tokens.front().position == 2);
        CPP_LLM_CHECK(suite, input.tokens.front().kv_write.slot == 0);
    });

    suite.run("failed prefill rolls back block reservation", [&] {
        GenerationConfig config;
        config.max_new_tokens = 1;
        Runtime runtime(config);
        GreedySampler sampler;
        Request request({1, 2, 3, 4});
        auto executor = std::make_shared<testing::ChunkedTokenExecutor>(6, true);
        chapter07_kv_cache::KvBlockPool pool(2, 2);

        const auto failed = ContinuousBatchScheduler{}.run_step(
            runtime, {{50, &request, &sampler, executor}}, 2, 0, {}, &pool);
        CPP_LLM_CHECK(suite, failed.generated_tokens == 0);
        CPP_LLM_CHECK(suite, request.failed());
        CPP_LLM_CHECK(suite, !pool.contains(50));
        CPP_LLM_CHECK(suite, pool.free_blocks() == 2);
        CPP_LLM_CHECK(
            suite, inspect_request_execution_state(50, request, executor.get(), &pool).consistent());
    });

    return suite.finish();
}
