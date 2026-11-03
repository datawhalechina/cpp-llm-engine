// Chapter 09：ServiceCore 请求级 CPU 并行测试。
// 职责：证明两个独立 Executor 的 Prefill 时间区间发生重叠且结果隔离。
// 验证入口：chapter09_parallel_dispatch_test 或带 parallel 标签的 CTest。
#include "chapter06_generation/runtime.h"
#include "chapter06_generation/sampler.h"
#include "chapter09_serving/serving_core.h"
#include "cpp_llm_engine/testing/test.h"
#include "support/service_test_support.h"

#include <memory>

int main() {
    using namespace chapter06_generation;
    using namespace chapter09_serving;
    using testing::CoordinatedExecutor;
    using testing::ParallelGate;
    cpp_llm_engine::testing::TestSuite suite("chapter09_parallel_dispatch");

    suite.run("independent executors overlap without mixing generated tokens", [&] {
        GenerationConfig config;
        config.max_new_tokens = 1;
        Runtime runtime(config);
        GreedySampler sampler;
        ServingCore service(2, 2);
        Request request_a({1});
        Request request_b({2});
        auto gate = std::make_shared<ParallelGate>();
        auto executor_a = std::make_shared<CoordinatedExecutor>(gate, 3);
        auto executor_b = std::make_shared<CoordinatedExecutor>(gate, 4);

        CPP_LLM_REQUIRE(suite, service.admit(35, request_a, sampler, executor_a).accepted());
        CPP_LLM_REQUIRE(suite, service.admit(36, request_b, sampler, executor_b).accepted());
        const auto stats = service.dispatch_parallel_requests(runtime, 2);
        CPP_LLM_CHECK_EQ(suite, stats.admitted, 2U);
        CPP_LLM_CHECK_EQ(suite, stats.completed, 2U);
        CPP_LLM_CHECK_EQ(suite, stats.generated_tokens, 2U);
        CPP_LLM_CHECK_EQ(suite, gate->peak, 2U);
        CPP_LLM_CHECK(suite, request_a.generated() == TokenSequence({3}));
        CPP_LLM_CHECK(suite, request_b.generated() == TokenSequence({4}));
        CPP_LLM_CHECK_EQ(suite, service.release_completed(), 2U);
    });

    return suite.finish();
}
