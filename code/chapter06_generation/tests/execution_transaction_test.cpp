// Chapter 06：执行输入预留和失败回滚事务测试。
// 职责：验证 Decode 写槽必须先预留，模型未推进时释放临时块。
// 验证入口：带 contract、transaction、cache 标签的 Chapter06 CTest。
#include "chapter06_generation/serving_preview.h"

#include "chapter06_generation/request.h"
#include "chapter06_generation/runtime.h"
#include "chapter06_generation/sampler.h"
#include "cpp_llm_engine/testing/test.h"
#include "support/serving_test_executors.h"

#include <memory>
#include <string>
#include <vector>

int main() {
    using namespace chapter06_generation;
    using testing::ChunkedTokenExecutor;
    cpp_llm_engine::testing::TestSuite suite("chapter06_execution_transaction");

    suite.run("decode batch input requires a reserved write slot", [&] {
        GenerationConfig config;
        config.max_new_tokens = 2;
        config.max_context_tokens = 16;
        Runtime runtime(config);
        GreedySampler sampler;
        Request request({1, 2});
        auto executor = std::make_shared<ChunkedTokenExecutor>(6);
        KvBlockPool pool(2, 2);
        const std::vector<ScheduledRequest> requests = {
            {45, &request, &sampler, executor},
        };
        const ContinuousBatchScheduler scheduler;
        const auto prefill = scheduler.run_step(runtime, requests, 2, 0, {}, &pool);
        CPP_LLM_CHECK(suite, prefill.generated_tokens == 1);
        CPP_LLM_CHECK(suite, !request.terminal());
        const auto after_prefill = inspect_request_execution_state(
            45, request, executor.get(), &pool);
        CPP_LLM_CHECK(suite, after_prefill.token_history_tokens == 3);
        CPP_LLM_CHECK(suite, after_prefill.model.committed_tokens == 2);
        CPP_LLM_CHECK(suite, after_prefill.block_table_tokens == 2);
        CPP_LLM_CHECK(suite, after_prefill.consistent());

        const auto plan = StepBatchPlanner{}.build(requests, 1);
        CPP_LLM_CHECK(suite, plan.decode_requests == 1);
        ExecutorBatchInput input;
        std::string error;
        CPP_LLM_CHECK(
            suite, !build_decode_executor_batch_input(plan, requests, pool, &input, &error));
        CPP_LLM_CHECK(suite, error == "decode batch has no reserved KV write slot");
        CPP_LLM_CHECK(suite, input.tokens.empty());

        CPP_LLM_CHECK(suite, pool.reserve(45, 3));
        CPP_LLM_CHECK(
            suite, build_decode_executor_batch_input(plan, requests, pool, &input, &error));
        CPP_LLM_CHECK(suite, error.empty());
        CPP_LLM_CHECK(suite, input.valid());
        CPP_LLM_REQUIRE(suite, input.tokens.size() == 1);
        CPP_LLM_CHECK(suite, input.tokens[0].request_id == 45);
        CPP_LLM_CHECK(suite, input.tokens[0].token == 6);
        CPP_LLM_CHECK(suite, input.tokens[0].position == 2);
        CPP_LLM_CHECK(suite, input.tokens[0].kv_write.block == 1);
        CPP_LLM_CHECK(suite, input.tokens[0].kv_write.slot == 0);

        const auto decode = scheduler.run_step(runtime, requests, 1, 0, {}, &pool);
        CPP_LLM_CHECK(suite, decode.generated_tokens == 1);
        CPP_LLM_CHECK(suite, request.finished());
        const auto after_decode = inspect_request_execution_state(
            45, request, executor.get(), &pool);
        CPP_LLM_CHECK(suite, after_decode.token_history_tokens == 4);
        CPP_LLM_CHECK(suite, after_decode.model.committed_tokens == 3);
        CPP_LLM_CHECK(suite, after_decode.block_table_tokens == 3);
        CPP_LLM_CHECK(suite, after_decode.consistent());
    });

    suite.run("failed prefill rolls back block reservation", [&] {
        GenerationConfig config;
        config.max_new_tokens = 1;
        config.max_context_tokens = 16;
        Runtime runtime(config);
        GreedySampler sampler;
        Request request({1, 2, 3, 4});
        auto executor = std::make_shared<ChunkedTokenExecutor>(6, true);
        KvBlockPool pool(2, 2);
        const ContinuousBatchScheduler scheduler;

        const auto failed = scheduler.run_step(
            runtime, {{50, &request, &sampler, executor}}, 2, 0, {}, &pool);
        CPP_LLM_CHECK(suite, failed.scheduled_requests == 1);
        CPP_LLM_CHECK(suite, failed.generated_tokens == 0);
        CPP_LLM_CHECK(suite, request.failed());
        CPP_LLM_CHECK(suite, request.prefill_offset() == 0);
        CPP_LLM_CHECK(suite, !pool.contains(50));
        CPP_LLM_CHECK(suite, pool.free_blocks() == 2);
        const auto state = inspect_request_execution_state(
            50, request, executor.get(), &pool);
        CPP_LLM_CHECK(suite, state.token_history_tokens == 0);
        CPP_LLM_CHECK(suite, state.model.committed_tokens == 0);
        CPP_LLM_CHECK(suite, !state.has_block_table);
        CPP_LLM_CHECK(suite, state.consistent());
    });

    return suite.finish();
}
