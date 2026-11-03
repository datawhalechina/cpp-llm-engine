// Chapter 09：部分 Prefill 请求的取消与 KV block 回收测试。
// 职责：验证取消终态清空请求 cache，并归还尚未完成请求占用的块。
// 验证入口：chapter09_cancellation_release_test 或带 cancellation、cache 标签的 CTest。
#include "chapter06_generation/runtime.h"
#include "chapter06_generation/sampler.h"
#include "chapter06_generation/serving_preview.h"
#include "chapter09_serving/serving_core.h"
#include "cpp_llm_engine/testing/test.h"
#include "support/service_test_support.h"

#include <memory>

int main() {
    using namespace chapter06_generation;
    using namespace chapter09_serving;
    using testing::ChunkedExecutor;
    cpp_llm_engine::testing::TestSuite suite("chapter09_cancellation_release");

    suite.run("cancelling partial prefill releases request and block state", [&] {
        GenerationConfig config;
        config.max_new_tokens = 2;
        Runtime runtime(config);
        GreedySampler sampler;
        auto pool = std::make_shared<KvBlockPool>(3, 2);
        ServingCore service(1, 1, ServiceNow{}, pool);
        Request request({1, 2, 3, 4, 5});
        auto executor = std::make_shared<ChunkedExecutor>();

        CPP_LLM_REQUIRE(suite, service.admit(40, request, sampler, executor).accepted());
        const auto partial = service.dispatch_step(runtime, 2);
        CPP_LLM_CHECK_EQ(suite, partial.generated_tokens, 0U);
        CPP_LLM_CHECK_EQ(suite, partial.scheduled_tokens, 2U);
        CPP_LLM_CHECK_EQ(suite, request.prefill_offset(), 2U);
        CPP_LLM_CHECK_EQ(suite, pool->token_count(40), 2U);
        CPP_LLM_CHECK(suite, service.cancel(40));
        CPP_LLM_CHECK_EQ(suite, service.release_completed(), 1U);
        CPP_LLM_CHECK(suite, request.cancelled());
        CPP_LLM_CHECK(suite, request.cache().empty());
        CPP_LLM_CHECK_EQ(suite, pool->free_blocks(), 3U);
        CPP_LLM_CHECK(suite, !pool->contains(40));
    });

    return suite.finish();
}
