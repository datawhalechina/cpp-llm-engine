// Chapter 09：ServiceCore 准入、同步执行和终态释放组件测试。
// 职责：验证容量、重复 ID、执行失败、超时和请求级资源回收。
// 验证入口：chapter09_serving_test 或带 serving、lifecycle 标签的 CTest。
#include "chapter06_generation/embedding.h"
#include "chapter06_generation/position_encoding.h"
#include "chapter06_generation/runtime.h"
#include "chapter06_generation/sampler.h"
#include "chapter06_generation/transformer.h"
#include "chapter09_serving/serving_core.h"
#include "cpp_llm_engine/testing/test.h"
#include "support/service_test_support.h"

#include <memory>
#include <stdexcept>

int main() {
    using namespace chapter06_generation;
    using namespace chapter09_serving;
    using testing::FailingExecutor;
    using testing::FixedTokenExecutor;
    cpp_llm_engine::testing::TestSuite suite("chapter09_serving_core");

    suite.run("configuration rejects a batch larger than service capacity", [&] {
        CPP_LLM_CHECK_THROWS(suite, std::invalid_argument, ServingCore(1, 2));
    });

    suite.run("admission records capacity duplicate and cancellation outcomes", [&] {
        GreedySampler sampler;
        Request request_a({1, 2});
        Request request_b({3, 4});
        Request request_c({5, 6});
        Request rejected({8});
        ServingCore service(3, 2);

        CPP_LLM_CHECK(suite, service.admit(1, request_a, sampler).accepted());
        CPP_LLM_CHECK(suite, service.admit(2, request_b, sampler).accepted());
        CPP_LLM_CHECK(suite, service.admit(3, request_c, sampler).accepted());
        CPP_LLM_CHECK(
            suite,
            service.admit(1, request_c, sampler).status == AdmissionStatus::DuplicateId);
        CPP_LLM_CHECK(
            suite,
            service.admit(4, rejected, sampler).status == AdmissionStatus::QueueFull);
        CPP_LLM_CHECK_EQ(suite, service.waiting_size(), 3U);
        CPP_LLM_CHECK(suite, service.cancel(3));
        CPP_LLM_CHECK(suite, request_c.cancelled());
        CPP_LLM_CHECK_EQ(suite, service.waiting_size(), 2U);
        CPP_LLM_CHECK_EQ(suite, service.metrics().accepted_total, 3U);
        CPP_LLM_CHECK_EQ(suite, service.metrics().duplicate_total, 1U);
        CPP_LLM_CHECK_EQ(suite, service.metrics().queue_full_total, 1U);
        CPP_LLM_CHECK_EQ(suite, service.metrics().cancelled_total, 1U);
        CPP_LLM_CHECK_EQ(suite, service.metrics().peak_inflight, 3U);
    });

    suite.run("synchronous dispatch completes and releases request state", [&] {
        GenerationConfig config;
        config.max_new_tokens = 2;
        config.max_context_tokens = 32;
        Runtime runtime(config, Transformer(2, Embedding(32), PositionEncoding(16)));
        GreedySampler sampler;
        Request request_a({1, 2});
        Request request_b({3, 4});
        ServingCore service(2, 2);

        CPP_LLM_REQUIRE(suite, service.admit(1, request_a, sampler).accepted());
        CPP_LLM_REQUIRE(suite, service.admit(2, request_b, sampler).accepted());
        const ServiceStats stats = service.dispatch(runtime);
        CPP_LLM_CHECK_EQ(suite, stats.admitted, 2U);
        CPP_LLM_CHECK_EQ(suite, stats.completed, 2U);
        CPP_LLM_CHECK_EQ(suite, stats.generated_tokens, 4U);
        CPP_LLM_CHECK(suite, request_a.finished() && request_b.finished());
        CPP_LLM_CHECK_EQ(suite, service.running_size(), 2U);
        CPP_LLM_CHECK_EQ(suite, service.release_completed(), 2U);
        CPP_LLM_CHECK(suite, request_a.cache().empty() && request_b.cache().empty());
        CPP_LLM_CHECK_EQ(suite, service.running_size(), 0U);
        CPP_LLM_CHECK_EQ(suite, service.metrics().released_total, 2U);
    });

    suite.run("per-request executors stay isolated and failures are observable", [&] {
        GenerationConfig config;
        config.max_new_tokens = 2;
        config.max_context_tokens = 32;
        Runtime runtime(config);
        GreedySampler sampler;
        ServingCore service(2, 2);
        Request request_a({1});
        Request request_b({2});
        auto executor_a = std::make_shared<FixedTokenExecutor>(3);
        auto executor_b = std::make_shared<FixedTokenExecutor>(4);

        CPP_LLM_REQUIRE(suite, service.admit(10, request_a, sampler, executor_a).accepted());
        CPP_LLM_REQUIRE(suite, service.admit(11, request_b, sampler, executor_b).accepted());
        const auto stats = service.dispatch(runtime);
        CPP_LLM_CHECK_EQ(suite, stats.completed, 2U);
        CPP_LLM_CHECK(suite, request_a.generated() == TokenSequence({3, 3}));
        CPP_LLM_CHECK(suite, request_b.generated() == TokenSequence({4, 4}));
        CPP_LLM_CHECK_EQ(suite, executor_a->prefill_calls, 1U);
        CPP_LLM_CHECK_EQ(suite, executor_a->decode_calls, 1U);
        CPP_LLM_CHECK_EQ(suite, executor_b->prefill_calls, 1U);
        CPP_LLM_CHECK_EQ(suite, executor_b->decode_calls, 1U);
        CPP_LLM_CHECK_EQ(suite, service.release_completed(), 2U);

        Request failed({5});
        auto failing_executor = std::make_shared<FailingExecutor>();
        CPP_LLM_REQUIRE(suite, service.admit(12, failed, sampler, failing_executor).accepted());
        CPP_LLM_CHECK_EQ(suite, service.dispatch(runtime).generated_tokens, 0U);
        CPP_LLM_CHECK(suite, failed.failed());
        CPP_LLM_CHECK(suite, failed.failure_reason() == "service model failure");
        CPP_LLM_CHECK_EQ(suite, service.release_completed(), 1U);
        CPP_LLM_CHECK_EQ(suite, service.metrics().failed_total, 1U);
    });

    suite.run("timeout removes a waiting request without dispatch", [&] {
        GenerationConfig config;
        Runtime runtime(config);
        GreedySampler sampler;
        Request request({7});
        ServingCore service(1, 1);

        CPP_LLM_REQUIRE(suite, service.admit(5, request, sampler).accepted());
        CPP_LLM_CHECK(suite, service.timeout(5));
        CPP_LLM_CHECK(suite, request.timed_out());
        CPP_LLM_CHECK_EQ(suite, service.metrics().timed_out_total, 1U);
        CPP_LLM_CHECK_EQ(suite, service.waiting_size(), 0U);
        CPP_LLM_CHECK_EQ(suite, service.dispatch(runtime).completed, 0U);
    });

    return suite.finish();
}
