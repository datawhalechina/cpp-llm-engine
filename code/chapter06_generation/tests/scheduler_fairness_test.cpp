// Chapter 06：Scheduler 饥饿保护和 Prefill 进度边界测试。
// 职责：验证最小预算下保留 Prefill 机会，并拒绝越界推进。
// 验证入口：带 contract、scheduler、fairness 标签的 Chapter06 CTest。
#include "chapter06_generation/serving_preview.h"

#include "chapter06_generation/request.h"
#include "chapter06_generation/runtime.h"
#include "chapter06_generation/sampler.h"
#include "cpp_llm_engine/testing/test.h"
#include "support/serving_test_executors.h"

#include <memory>
#include <stdexcept>

int main() {
    using namespace chapter06_generation;
    using testing::ChunkedTokenExecutor;
    using testing::FixedTokenExecutor;
    cpp_llm_engine::testing::TestSuite suite("chapter06_scheduler_fairness");

    suite.run("one token budget reserves progress for waiting prefill", [&] {
        GenerationConfig config;
        config.max_new_tokens = 3;
        Runtime runtime(config);
        GreedySampler sampler;
        Request decoding_request({1});
        auto decoding_executor = std::make_shared<FixedTokenExecutor>(3);
        CPP_LLM_CHECK(
            suite, runtime.decode_one(decoding_request, sampler, *decoding_executor) == 1);
        Request waiting_prefill({2, 3, 4, 5});
        auto waiting_executor = std::make_shared<ChunkedTokenExecutor>(4);
        const std::size_t generated_before = decoding_request.generated_size();

        const auto stats = ContinuousBatchScheduler{}.run_step(
            runtime,
            {
                {60, &decoding_request, &sampler, decoding_executor},
                {61, &waiting_prefill, &sampler, waiting_executor},
            },
            1,
            1);
        CPP_LLM_CHECK(suite, stats.prefill_requests == 1);
        CPP_LLM_CHECK(suite, stats.decode_requests == 0);
        CPP_LLM_CHECK(suite, decoding_request.generated_size() == generated_before);
        CPP_LLM_CHECK(suite, waiting_prefill.prefill_offset() == 1);
    });

    suite.run("prefill progress cannot exceed prompt length", [&] {
        bool rejected = false;
        try {
            Request request({1});
            request.advance_prefill(2);
        } catch (const std::out_of_range&) {
            rejected = true;
        }
        CPP_LLM_CHECK(suite, rejected);
    });

    return suite.finish();
}
