// Chapter 09：ServiceCore 混合终态序列测试。
// 职责：在同一服务实例中交错成功、模型失败、取消和超时，并核对资源与指标。
// 验证入口：chapter09_fault_sequence_test 或带 fault、sequence 标签的 CTest。
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
    using testing::FailingExecutor;

    cpp_llm_engine::testing::TestSuite suite("chapter09_fault_sequence");
    suite.run("混合终态保持请求、块池和指标一致", [&] {
        GenerationConfig config;
        config.max_new_tokens = 1;
        config.max_context_tokens = 8;
        Runtime runtime(config);
        GreedySampler sampler;
        auto pool = std::make_shared<KvBlockPool>(4, 2);
        ServingCore service(4, 2, ServiceNow{}, pool);

        Request successful({1});
        Request failed({2});
        Request cancelled({3});
        Request timed_out({4});
        CPP_LLM_REQUIRE(suite, service.admit(
            100, successful, sampler, std::make_shared<ChunkedExecutor>()).accepted());
        CPP_LLM_REQUIRE(suite, service.admit(
            101, failed, sampler, std::make_shared<FailingExecutor>()).accepted());
        CPP_LLM_REQUIRE(suite, service.admit(
            102, cancelled, sampler, std::make_shared<ChunkedExecutor>()).accepted());
        CPP_LLM_REQUIRE(suite, service.admit(
            103, timed_out, sampler, std::make_shared<ChunkedExecutor>()).accepted());

        CPP_LLM_CHECK(suite, service.cancel(102));
        CPP_LLM_CHECK(suite, service.timeout(103));
        CPP_LLM_CHECK_EQ(suite, service.waiting_size(), 2U);

        const auto step = service.dispatch_step(runtime, 2);
        CPP_LLM_CHECK_EQ(suite, step.scheduled_requests, 2U);
        CPP_LLM_CHECK_EQ(suite, step.generated_tokens, 1U);
        CPP_LLM_CHECK(suite, successful.finished());
        CPP_LLM_CHECK(suite, failed.failed());
        CPP_LLM_CHECK(suite, cancelled.cancelled());
        CPP_LLM_CHECK(suite, timed_out.timed_out());
        CPP_LLM_CHECK_EQ(suite, service.release_completed(), 2U);

        const auto& metrics = service.metrics();
        CPP_LLM_CHECK_EQ(suite, service.inflight_size(), 0U);
        CPP_LLM_CHECK_EQ(suite, pool->free_blocks(), pool->total_blocks());
        CPP_LLM_CHECK_EQ(suite, metrics.accepted_total, 4U);
        CPP_LLM_CHECK_EQ(suite, metrics.completed_total, 1U);
        CPP_LLM_CHECK_EQ(suite, metrics.failed_total, 1U);
        CPP_LLM_CHECK_EQ(suite, metrics.cancelled_total, 1U);
        CPP_LLM_CHECK_EQ(suite, metrics.timed_out_total, 1U);
        CPP_LLM_CHECK_EQ(suite, metrics.released_total, 2U);
        CPP_LLM_CHECK_EQ(suite, metrics.generated_tokens_total, 1U);
    });
    return suite.finish();
}
