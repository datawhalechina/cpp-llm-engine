// Chapter 06：BlockPool 容量和调度协作测试。
// 职责：验证块地址、容量不足时的推迟以及释放后的复用。
// 验证入口：带 component、scheduler、cache 标签的 Chapter06 CTest。
#include "chapter06_generation/serving_preview.h"

#include "chapter06_generation/request.h"
#include "chapter06_generation/runtime.h"
#include "chapter06_generation/sampler.h"
#include "cpp_llm_engine/testing/test.h"
#include "support/serving_test_executors.h"

#include <limits>
#include <memory>
#include <stdexcept>

int main() {
    using namespace chapter06_generation;
    using testing::FixedTokenExecutor;
    cpp_llm_engine::testing::TestSuite suite("chapter06_block_pool_scheduling");

    suite.run("invalid block pool dimensions are rejected", [&] {
        bool invalid_table = false;
        bool invalid_pool = false;
        bool overflow_pool = false;
        try { BlockTable table(0); } catch (const std::invalid_argument&) {
            invalid_table = true;
        }
        try { KvBlockPool pool(0, 2); } catch (const std::invalid_argument&) {
            invalid_pool = true;
        }
        try {
            KvBlockPool pool(2, std::numeric_limits<std::size_t>::max());
        } catch (const std::overflow_error&) {
            overflow_pool = true;
        }
        CPP_LLM_CHECK(suite, invalid_table);
        CPP_LLM_CHECK(suite, invalid_pool);
        CPP_LLM_CHECK(suite, overflow_pool);
    });

    suite.run("block table locates committed and reserved slots", [&] {
        KvBlockPool pool(3, 2);
        CPP_LLM_CHECK(suite, pool.total_blocks() == 3 && pool.free_blocks() == 3);
        CPP_LLM_CHECK(suite, pool.reserve(20, 3));
        CPP_LLM_CHECK(suite, pool.block_count(20) == 2);
        CPP_LLM_CHECK(suite, pool.commit_tokens(20, 3));
        CPP_LLM_CHECK(suite, pool.locate(20, 0).block == 0);
        CPP_LLM_CHECK(suite, pool.locate(20, 0).slot == 0);
        CPP_LLM_CHECK(suite, pool.locate(20, 2).block == 1);
        CPP_LLM_CHECK(suite, pool.locate(20, 2).slot == 0);
        bool rejected = false;
        try { (void)pool.locate(20, 3); } catch (const std::out_of_range&) {
            rejected = true;
        }
        CPP_LLM_CHECK(suite, rejected);
        CPP_LLM_CHECK(suite, pool.locate_reserved(20, 3).block == 1);
        CPP_LLM_CHECK(suite, pool.locate_reserved(20, 3).slot == 1);
        CPP_LLM_CHECK(suite, !pool.reserve(21, 3));
        CPP_LLM_CHECK(suite, !pool.contains(21));
        CPP_LLM_CHECK(suite, pool.reserve(21, 2));
        CPP_LLM_CHECK(suite, pool.commit_tokens(21, 2));
        CPP_LLM_CHECK(suite, pool.release(20));
        CPP_LLM_CHECK(suite, pool.reserve(21, 3));
        CPP_LLM_CHECK(suite, pool.trim_to_tokens(21, 2));
        CPP_LLM_CHECK(suite, pool.block_count(21) == 1);
        CPP_LLM_CHECK(suite, pool.release(21));
        CPP_LLM_CHECK(suite, pool.free_blocks() == 3);
        CPP_LLM_CHECK(suite, !pool.release(99));
    });

    suite.run("block pressure defers request and release admits it", [&] {
        GenerationConfig config;
        config.max_new_tokens = 1;
        config.max_context_tokens = 16;
        Runtime runtime(config);
        GreedySampler sampler;
        Request request_a({1, 2, 3});
        Request request_b({4, 5});
        auto executor_a = std::make_shared<FixedTokenExecutor>(5);
        auto executor_b = std::make_shared<FixedTokenExecutor>(6);
        const std::vector<ScheduledRequest> requests = {
            {30, &request_a, &sampler, executor_a},
            {31, &request_b, &sampler, executor_b},
        };
        KvBlockPool pool(2, 2);
        const ContinuousBatchScheduler scheduler;

        const auto limited = scheduler.run_step(runtime, requests, 5, 0, {}, &pool);
        CPP_LLM_CHECK(suite, limited.scheduled_requests == 1);
        CPP_LLM_CHECK(suite, limited.scheduled_tokens == 3);
        CPP_LLM_CHECK(suite, limited.deferred_requests == 1);
        CPP_LLM_CHECK(suite, request_a.finished());
        CPP_LLM_CHECK(suite, !request_b.prefilled());
        CPP_LLM_CHECK(suite, executor_b->prefill_calls == 0);
        CPP_LLM_CHECK(suite, pool.used_blocks() == 2);
        CPP_LLM_CHECK(suite, pool.release(30));

        const auto admitted = scheduler.run_step(runtime, requests, 2, 0, {}, &pool);
        CPP_LLM_CHECK(suite, admitted.scheduled_requests == 1);
        CPP_LLM_CHECK(suite, request_b.finished());
        CPP_LLM_CHECK(suite, pool.locate(31, 0).block == 0);
        CPP_LLM_CHECK(suite, pool.release(31));
        CPP_LLM_CHECK(suite, pool.free_blocks() == 2);
    });

    return suite.finish();
}
