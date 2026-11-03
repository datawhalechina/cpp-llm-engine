// Chapter 09：ServiceCore token/block budget 步进测试。
// 职责：验证动态补位、KV block 容量约束和 legacy dispatch 防误用。
// 验证入口：chapter09_step_dispatch_test 或带 scheduler、cache 标签的 CTest。
#include "chapter06_generation/runtime.h"
#include "chapter06_generation/sampler.h"
#include "chapter06_generation/serving_preview.h"
#include "chapter09_serving/serving_core.h"
#include "cpp_llm_engine/testing/test.h"
#include "support/service_test_support.h"

#include <memory>
#include <stdexcept>

int main() {
    using namespace chapter06_generation;
    using namespace chapter09_serving;
    using testing::FixedTokenExecutor;
    cpp_llm_engine::testing::TestSuite suite("chapter09_step_dispatch");

    suite.run("block-aware service requires step dispatch", [&] {
        GenerationConfig config;
        config.max_new_tokens = 2;
        Runtime runtime(config);
        GreedySampler sampler;
        auto pool = std::make_shared<KvBlockPool>(2, 2);
        ServingCore service(2, 2, ServiceNow{}, pool);
        Request request({1});
        auto executor = std::make_shared<FixedTokenExecutor>(3);
        CPP_LLM_REQUIRE(suite, service.admit(30, request, sampler, executor).accepted());
        CPP_LLM_CHECK_THROWS(suite, std::logic_error, service.dispatch(runtime));
    });

    suite.run("completed requests release blocks before waiting request enters", [&] {
        GenerationConfig config;
        config.max_new_tokens = 2;
        Runtime runtime(config);
        GreedySampler sampler;
        auto pool = std::make_shared<KvBlockPool>(2, 2);
        ServingCore service(3, 2, ServiceNow{}, pool);
        Request request_a({1});
        Request request_b({2});
        Request request_c({3});
        auto executor_a = std::make_shared<FixedTokenExecutor>(3);
        auto executor_b = std::make_shared<FixedTokenExecutor>(4);
        auto executor_c = std::make_shared<FixedTokenExecutor>(5);

        CPP_LLM_REQUIRE(suite, service.admit(30, request_a, sampler, executor_a).accepted());
        CPP_LLM_REQUIRE(suite, service.admit(31, request_b, sampler, executor_b).accepted());
        const auto first = service.dispatch_step(runtime, 2);
        CPP_LLM_CHECK_EQ(suite, first.admitted, 2U);
        CPP_LLM_CHECK_EQ(suite, first.generated_tokens, 2U);
        CPP_LLM_CHECK_EQ(suite, first.completed, 0U);
        CPP_LLM_CHECK_EQ(suite, first.scheduled_requests, 2U);
        CPP_LLM_CHECK_EQ(suite, first.scheduled_tokens, 2U);
        CPP_LLM_CHECK_EQ(suite, pool->free_blocks(), 0U);

        CPP_LLM_REQUIRE(suite, service.admit(32, request_c, sampler, executor_c).accepted());
        CPP_LLM_CHECK_EQ(suite, service.waiting_size(), 1U);
        const auto second = service.dispatch_step(runtime, 2);
        CPP_LLM_CHECK_EQ(suite, second.generated_tokens, 2U);
        CPP_LLM_CHECK_EQ(suite, second.completed, 2U);
        CPP_LLM_CHECK_EQ(suite, service.waiting_size(), 1U);
        CPP_LLM_CHECK_EQ(suite, service.release_completed(), 2U);
        CPP_LLM_CHECK_EQ(suite, pool->free_blocks(), 2U);

        const auto replacement = service.dispatch_step(runtime, 1);
        CPP_LLM_CHECK_EQ(suite, replacement.admitted, 1U);
        CPP_LLM_CHECK_EQ(suite, replacement.generated_tokens, 1U);
        CPP_LLM_CHECK_EQ(suite, service.waiting_size(), 0U);
        CPP_LLM_CHECK(suite, !request_c.terminal());
        CPP_LLM_CHECK_EQ(suite, service.dispatch_step(runtime, 1).completed, 1U);
        CPP_LLM_CHECK_EQ(suite, service.release_completed(), 1U);
        CPP_LLM_CHECK_EQ(suite, pool->free_blocks(), 2U);
        CPP_LLM_CHECK_EQ(suite, service.metrics().generated_tokens_total, 6U);
        CPP_LLM_CHECK_EQ(suite, service.metrics().completed_total, 3U);
    });

    return suite.finish();
}
