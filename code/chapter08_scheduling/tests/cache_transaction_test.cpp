// Chapter 08：验证调度、模型状态与 Chapter07 BlockTable 的跨章事务。
// 职责：检查预留槽位、提交计数和执行失败回滚。
// 验证入口：chapter08_cache_transaction_test。
#include "chapter08_scheduling/step_scheduler.h"

#include "chapter06_generation/request.h"
#include "chapter06_generation/runtime.h"
#include "chapter06_generation/sampler.h"
#include "chapter07_kv_cache/block_table.h"
#include "chapter07_kv_cache/paged_kv_cache.h"
#include "cpp_llm_engine/testing/test.h"
#include "cpp_llm_engine/testing/fault_injector.h"
#include "support/test_executors.h"

#include <memory>
#include <string>
#include <vector>

namespace {

class PartialPagedWriteFailure final
    : public chapter06_generation::ExecutorBatchRunner {
public:
    explicit PartialPagedWriteFailure(
        chapter07_kv_cache::CpuPagedKvCache* cache,
        cpp_llm_engine::testing::FaultInjector* faults)
        : cache_(cache), faults_(faults) {}

    bool decode_batch(
        const std::vector<std::shared_ptr<chapter06_generation::RequestModelExecutor>>&,
        const chapter06_generation::TokenSequence&,
        const std::vector<std::size_t>& expected_positions,
        const std::vector<std::size_t>& request_ids,
        std::vector<chapter06_generation::Logits>*,
        std::string* error) const override {
        const float key[] = {1.0f, 0.0f};
        const float value[] = {10.0f, 1.0f};
        if (cache_ == nullptr || expected_positions.size() != 1 ||
            request_ids.size() != 1 ||
            !cache_->write_token(
                request_ids.front(),
                0,
                expected_positions.front(),
                key,
                value,
                2,
                error)) {
            return false;
        }
        if (faults_ != nullptr && faults_->hit("paged_kv.after_write")) {
            if (error != nullptr) *error = "synthetic paged decode failure";
            return false;
        }
        return true;
    }

private:
    chapter07_kv_cache::CpuPagedKvCache* cache_;
    cpp_llm_engine::testing::FaultInjector* faults_;
};

class CountingBatchRunner final : public chapter06_generation::ExecutorBatchRunner {
public:
    bool decode_batch(
        const std::vector<std::shared_ptr<chapter06_generation::RequestModelExecutor>>&,
        const chapter06_generation::TokenSequence&,
        const std::vector<std::size_t>&,
        const std::vector<std::size_t>&,
        std::vector<chapter06_generation::Logits>*,
        std::string*) const override {
        ++calls;
        return true;
    }

    mutable std::size_t calls = 0;
};

void write_all_layers(
    cpp_llm_engine::testing::TestSuite& suite,
    chapter07_kv_cache::CpuPagedKvCache& cache,
    std::size_t request_id,
    std::size_t token) {
    const float key[] = {1.0f, 0.0f};
    const float value[] = {10.0f, 1.0f};
    for (std::size_t layer = 0; layer < cache.layer_count(); ++layer) {
        CPP_LLM_CHECK(
            suite,
            cache.write_token(request_id, layer, token, key, value, 2));
    }
}

}  // namespace

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
        cpp_llm_engine::testing::FaultInjector faults;
        faults.arm("executor.prefill_chunk");
        auto executor = std::make_shared<testing::ChunkedTokenExecutor>(6, &faults);
        chapter07_kv_cache::KvBlockPool pool(2, 2);

        const auto failed = ContinuousBatchScheduler{}.run_step(
            runtime, {{50, &request, &sampler, executor}}, 2, 0, {}, &pool);
        CPP_LLM_CHECK(suite, failed.generated_tokens == 0);
        CPP_LLM_CHECK(suite, request.failed());
        CPP_LLM_CHECK(suite, !pool.contains(50));
        CPP_LLM_CHECK(suite, pool.free_blocks() == 2);
        CPP_LLM_CHECK_EQ(suite, faults.calls("executor.prefill_chunk"), 1U);
        CPP_LLM_CHECK_EQ(suite, faults.failures("executor.prefill_chunk"), 1U);
        CPP_LLM_CHECK(
            suite, inspect_request_execution_state(50, request, executor.get(), &pool).consistent());
    });

    suite.run("invalid decode input preserves the previous batch", [&] {
        GenerationConfig config;
        config.max_new_tokens = 2;
        Runtime runtime(config);
        GreedySampler sampler;
        Request request({1, 2});
        auto executor = std::make_shared<testing::ChunkedTokenExecutor>(6);
        chapter07_kv_cache::KvBlockPool pool(2, 2);
        const std::vector<ScheduledRequest> requests{{60, &request, &sampler, executor}};
        const ContinuousBatchScheduler scheduler;
        CPP_LLM_CHECK(
            suite, scheduler.run_step(runtime, requests, 2, 0, {}, &pool).generated_tokens == 1);

        const StepBatchPlan plan = StepBatchPlanner{}.build(requests, 1);
        ExecutorBatchInput input;
        input.tokens.push_back({99, 99, 99, 99, {1, 1}});
        std::string error;
        CPP_LLM_CHECK(
            suite, !build_decode_executor_batch_input(plan, requests, pool, &input, &error));
        CPP_LLM_REQUIRE(suite, input.tokens.size() == 1);
        CPP_LLM_CHECK(suite, input.tokens.front().request_id == 99);
        CPP_LLM_CHECK(suite, error == "decode batch has no reserved KV write slot");
    });

    suite.run("duplicate request IDs are rejected before model execution", [&] {
        GenerationConfig config;
        config.max_new_tokens = 2;
        Runtime runtime(config);
        GreedySampler sampler;
        Request request_a({1});
        Request request_b({2});
        auto executor_a = std::make_shared<testing::FixedTokenExecutor>(5);
        auto executor_b = std::make_shared<testing::FixedTokenExecutor>(6);
        CPP_LLM_REQUIRE(
            suite,
            runtime.prefill_chunk(request_a, sampler, *executor_a, 1).generated_tokens == 1);
        CPP_LLM_REQUIRE(
            suite,
            runtime.prefill_chunk(request_b, sampler, *executor_b, 1).generated_tokens == 1);

        chapter07_kv_cache::KvBlockPool pool(2, 2);
        CPP_LLM_REQUIRE(suite, pool.reserve(65, 1));
        CPP_LLM_REQUIRE(suite, pool.commit_tokens(65, 1));
        CountingBatchRunner batch_runner;

        const auto failed = ContinuousBatchScheduler{}.run_step(
            runtime,
            {{65, &request_a, &sampler, executor_a},
             {65, &request_b, &sampler, executor_b}},
            2,
            0,
            {},
            &pool,
            0,
            nullptr,
            &batch_runner);

        CPP_LLM_CHECK(suite, batch_runner.calls == 0);
        CPP_LLM_CHECK(suite, failed.generated_tokens == 0);
        CPP_LLM_CHECK(suite, failed.model_batch_calls == 0);
        CPP_LLM_CHECK(suite, request_a.failed() && request_b.failed());
        CPP_LLM_CHECK(suite, pool.token_count(65) == 1);
        CPP_LLM_CHECK(suite, pool.capacity_tokens(65) == 2);
    });

    suite.run("failed scheduler step clears partial paged payload", [&] {
        GenerationConfig config;
        config.max_new_tokens = 2;
        config.max_context_tokens = 16;
        Runtime runtime(config);
        GreedySampler sampler;
        Request request({1});
        auto executor = std::make_shared<testing::FixedTokenExecutor>(6);
        CPP_LLM_REQUIRE(
            suite,
            runtime.prefill_chunk(request, sampler, *executor, 1).generated_tokens == 1);

        chapter07_kv_cache::CpuPagedKvCache cache(2, 2, 1, 1, 2);
        CPP_LLM_REQUIRE(suite, cache.reserve(70, 1));
        write_all_layers(suite, cache, 70, 0);
        CPP_LLM_REQUIRE(suite, cache.commit_tokens(70, 1));
        cpp_llm_engine::testing::FaultInjector faults;
        faults.arm("paged_kv.after_write");
        PartialPagedWriteFailure batch_runner(&cache, &faults);

        const auto failed = ContinuousBatchScheduler{}.run_step(
            runtime,
            {{70, &request, &sampler, executor}},
            1,
            0,
            {},
            &cache,
            0,
            nullptr,
            &batch_runner);
        CPP_LLM_CHECK(suite, failed.generated_tokens == 0);
        CPP_LLM_CHECK(suite, request.failed());
        CPP_LLM_CHECK(suite, cache.token_count(70) == 1);
        CPP_LLM_CHECK(suite, cache.block_count(70) == 1);
        CPP_LLM_CHECK_EQ(suite, faults.calls("paged_kv.after_write"), 1U);
        CPP_LLM_CHECK_EQ(suite, faults.failures("paged_kv.after_write"), 1U);

        CPP_LLM_REQUIRE(suite, cache.reserve(71, 1));
        const float key[] = {0.0f, 1.0f};
        const float value[] = {2.0f, 20.0f};
        CPP_LLM_REQUIRE(suite, cache.write_token(71, 1, 0, key, value, 2));
        std::string error;
        CPP_LLM_CHECK(suite, !cache.commit_tokens(71, 1, &error));
        CPP_LLM_CHECK(suite, error.find("every layer") != std::string::npos);
    });

    return suite.finish();
}
