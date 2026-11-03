// Chapter 09：ServiceCore 在线延迟指标测试。
// 职责：使用手动单调时钟固定 queue wait、TTFT 和 completion 口径。
// 验证入口：chapter09_service_metrics_test 或带 metrics、latency 标签的 CTest。
#include "chapter06_generation/runtime.h"
#include "chapter06_generation/sampler.h"
#include "chapter09_serving/serving_core.h"
#include "cpp_llm_engine/testing/test.h"
#include "support/service_test_support.h"

#include <chrono>
#include <cmath>
#include <memory>

namespace {

bool near(double left, double right) {
    return std::abs(left - right) < 1e-12;
}

}  // namespace

int main() {
    using namespace chapter06_generation;
    using namespace chapter09_serving;
    using testing::FixedTokenExecutor;
    using testing::ManualClock;
    cpp_llm_engine::testing::TestSuite suite("chapter09_service_metrics");

    suite.run("manual clock fixes queue wait TTFT and completion sums", [&] {
        GenerationConfig config;
        config.max_new_tokens = 2;
        Runtime runtime(config);
        GreedySampler sampler;
        ManualClock clock;
        ServingCore service(2, 1, [&clock]() { return clock.now(); });
        Request request_a({1});
        Request request_b({2});
        auto executor_a = std::make_shared<FixedTokenExecutor>(3);
        auto executor_b = std::make_shared<FixedTokenExecutor>(4);

        CPP_LLM_REQUIRE(suite, service.admit(20, request_a, sampler, executor_a).accepted());
        clock.advance(std::chrono::milliseconds(4));
        CPP_LLM_REQUIRE(suite, service.admit(21, request_b, sampler, executor_b).accepted());
        clock.advance(std::chrono::milliseconds(6));
        const auto advance_after_token = [&clock](std::size_t, TokenId, std::size_t) {
            clock.advance(std::chrono::milliseconds(3));
        };
        CPP_LLM_CHECK_EQ(suite, service.dispatch(runtime, advance_after_token).completed, 1U);
        CPP_LLM_CHECK_EQ(suite, service.release_completed(), 1U);
        CPP_LLM_CHECK_EQ(suite, service.dispatch(runtime, advance_after_token).completed, 1U);
        CPP_LLM_CHECK_EQ(suite, service.release_completed(), 1U);

        const auto& metrics = service.metrics();
        CPP_LLM_CHECK_EQ(suite, metrics.queue_wait_count, 2U);
        CPP_LLM_CHECK_EQ(suite, metrics.ttft_count, 2U);
        CPP_LLM_CHECK_EQ(suite, metrics.completion_count, 2U);
        CPP_LLM_CHECK(suite, near(metrics.queue_wait_seconds_sum, 0.022));
        CPP_LLM_CHECK(suite, near(metrics.ttft_seconds_sum, 0.022));
        CPP_LLM_CHECK(suite, near(metrics.completion_seconds_sum, 0.028));
    });

    return suite.finish();
}
