// Chapter 08：验证 Round-robin 与教学 Tensor Batch 调度基线。
// 职责：比较逐请求结果、调用计数和流式事件归属。
// 验证入口：chapter08_scheduler_test。
#include "chapter08_scheduling/scheduler.h"

#include "chapter06_generation/embedding.h"
#include "chapter06_generation/position_encoding.h"
#include "chapter06_generation/request.h"
#include "chapter06_generation/runtime.h"
#include "chapter06_generation/sampler.h"
#include "chapter06_generation/transformer.h"
#include "cpp_llm_engine/testing/test.h"
#include "support/test_executors.h"

#include <memory>
#include <utility>
#include <vector>

int main() {
    using namespace chapter06_generation;
    using namespace chapter08_scheduling;
    cpp_llm_engine::testing::TestSuite suite("chapter08_scheduler");

    suite.run("round robin skips terminal requests and preserves order", [&] {
        Embedding embedding(32);
        PositionEncoding positions(16);
        Transformer transformer(2, embedding, positions);
        GenerationConfig config;
        config.max_new_tokens = 2;
        Runtime runtime(config, transformer);
        GreedySampler sampler;
        Request request_a({1, 2, 3});
        Request request_b({4, 5});
        Request cancelled({6});
        cancelled.cancel();
        std::vector<std::pair<std::size_t, std::size_t>> streamed;

        const auto stats = RoundRobinScheduler{}.run_with_stats(
            runtime,
            {{0, &request_a, &sampler, {}},
             {1, &request_b, &sampler, {}},
             {2, &cancelled, &sampler, {}}},
            [&](std::size_t request_id, TokenId, std::size_t step) {
                streamed.emplace_back(request_id, step);
            });

        CPP_LLM_CHECK(suite, stats.scheduled_requests == 3);
        CPP_LLM_CHECK(suite, stats.prefetched_requests == 2);
        CPP_LLM_CHECK(suite, stats.generated_tokens == 4);
        CPP_LLM_CHECK(suite, stats.completed_requests == 2);
        const std::vector<std::pair<std::size_t, std::size_t>> expected{
            {0, 0}, {1, 0}, {0, 1}, {1, 1},
        };
        CPP_LLM_CHECK(suite, streamed == expected);
        CPP_LLM_CHECK(suite, cancelled.cache().empty());
    });

    suite.run("round robin keeps request executors isolated", [&] {
        GenerationConfig config;
        config.max_new_tokens = 2;
        Runtime runtime(config);
        GreedySampler sampler;
        Request request_a({1, 2});
        Request request_b({3});
        auto executor_a = std::make_shared<testing::FixedTokenExecutor>(5);
        auto executor_b = std::make_shared<testing::FixedTokenExecutor>(6);
        std::vector<std::size_t> stream_ids;
        const auto stats = RoundRobinScheduler{}.run_with_stats(
            runtime,
            {{10, &request_a, &sampler, executor_a}, {11, &request_b, &sampler, executor_b}},
            [&](std::size_t request_id, TokenId, std::size_t) {
                stream_ids.push_back(request_id);
            });
        CPP_LLM_CHECK(suite, stats.generated_tokens == 4);
        CPP_LLM_CHECK(suite, stats.completed_requests == 2);
        CPP_LLM_CHECK(suite, request_a.generated() == TokenSequence({5, 5}));
        CPP_LLM_CHECK(suite, request_b.generated() == TokenSequence({6, 6}));
        CPP_LLM_CHECK(suite, stream_ids == std::vector<std::size_t>({10, 11, 10, 11}));
    });

    suite.run("tensor batch matches round robin output", [&] {
        Embedding embedding(64);
        PositionEncoding positions(32);
        Transformer transformer(2, embedding, positions);
        GenerationConfig config;
        config.max_new_tokens = 3;
        config.max_context_tokens = 16;
        GreedySampler sampler;

        Request sequential_a({1, 2, 3});
        Request sequential_b({4, 5});
        Runtime sequential_runtime(config, transformer);
        RoundRobinScheduler{}.run(
            sequential_runtime,
            {{20, &sequential_a, &sampler, {}}, {21, &sequential_b, &sampler, {}}});

        Request batch_a({1, 2, 3});
        Request batch_b({4, 5});
        Runtime batch_runtime(config, transformer);
        const auto stats = TensorBatchScheduler{}.run_with_stats(
            batch_runtime,
            {{20, &batch_a, &sampler, {}}, {21, &batch_b, &sampler, {}}});
        CPP_LLM_CHECK(suite, stats.model_batch_calls == 4);
        CPP_LLM_CHECK(suite, stats.tensor_batch_steps == 3);
        CPP_LLM_CHECK(suite, batch_a.generated() == sequential_a.generated());
        CPP_LLM_CHECK(suite, batch_b.generated() == sequential_b.generated());
    });

    return suite.finish();
}
