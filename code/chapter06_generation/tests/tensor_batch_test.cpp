// Chapter 06：连续 Ragged Tensor Batch 命名组件测试。
// 职责：验证批量 Prefill/Decode、请求提交、调用计数和逐请求等价性。
// 验证入口：带 component、batch 标签的 Chapter06 CTest。
#include "chapter06_generation/runtime.h"
#include "chapter06_generation/sampler.h"
#include "chapter06_generation/scheduler.h"
#include "chapter06_generation/transformer.h"

#include "cpp_llm_engine/testing/test.h"

#include <cstddef>
#include <utility>
#include <vector>

int main() {
    using namespace chapter06_generation;
    cpp_llm_engine::testing::TestSuite suite("chapter06_tensor_batch");

    suite.run("embedding batch produces valid ragged offsets", [&] {
        Embedding embedding(64);
        const RaggedBatch embedded = embedding.forward_batch({{1, 2, 3}, {4, 5}});
        CPP_LLM_CHECK(suite, embedded.valid());
        CPP_LLM_CHECK(suite, embedded.batch_size() == 2);
        CPP_LLM_CHECK(
            suite, embedded.row_offsets == std::vector<std::size_t>({0, 3, 5}));
    });

    suite.run("batched transformer prefill matches individual calls", [&] {
        Embedding embedding(64);
        PositionEncoding position_encoding(32);
        Transformer transformer(2, embedding, position_encoding);
        Request request_a({1, 2, 3});
        Request request_b({4, 5});
        const auto batch = transformer.prefill_batch({&request_a, &request_b});
        CPP_LLM_REQUIRE(suite, batch.size() == 2);
        CPP_LLM_CHECK(suite, batch[0] == transformer.prefill(request_a));
        CPP_LLM_CHECK(suite, batch[1] == transformer.prefill(request_b));
    });

    suite.run("duplicate request is rejected without partial mutation", [&] {
        Embedding embedding(64);
        PositionEncoding position_encoding(32);
        Transformer transformer(2, embedding, position_encoding);
        GenerationConfig config;
        config.max_new_tokens = 2;
        Runtime runtime(config, transformer);
        Request request({7, 8});

        const auto result = runtime.prefill_batch({&request, &request});
        CPP_LLM_CHECK(suite, result.model_calls == 0);
        CPP_LLM_CHECK(suite, !request.prefilled());
        CPP_LLM_CHECK(suite, request.cache().empty());
    });

    suite.run("tensor batch scheduler matches round robin output", [&] {
        Embedding embedding(64);
        PositionEncoding position_encoding(32);
        Transformer transformer(2, embedding, position_encoding);
        GenerationConfig config;
        config.max_new_tokens = 3;
        config.max_context_tokens = 16;
        config.use_kv_cache = true;
        GreedySampler sampler;

        Request sequential_a({1, 2, 3});
        Request sequential_b({4, 5});
        Runtime sequential_runtime(config, transformer);
        const auto sequential_stats = RoundRobinScheduler{}.run_with_stats(
            sequential_runtime,
            {
                {10, &sequential_a, &sampler, {}},
                {11, &sequential_b, &sampler, {}},
            });
        CPP_LLM_CHECK(suite, sequential_stats.completed_requests == 2);
        CPP_LLM_CHECK(suite, sequential_stats.generated_tokens == 6);

        Request batch_a({1, 2, 3});
        Request batch_b({4, 5});
        Runtime batch_runtime(config, transformer);
        std::vector<std::pair<std::size_t, std::size_t>> streamed_steps;
        const auto batch_stats = TensorBatchScheduler{}.run_with_stats(
            batch_runtime,
            {
                {10, &batch_a, &sampler, {}},
                {11, &batch_b, &sampler, {}},
            },
            [&](std::size_t request_id, TokenId, std::size_t step) {
                streamed_steps.emplace_back(request_id, step);
            });

        CPP_LLM_CHECK(suite, batch_stats.scheduled_requests == 2);
        CPP_LLM_CHECK(suite, batch_stats.prefetched_requests == 2);
        CPP_LLM_CHECK(suite, batch_stats.generated_tokens == 6);
        CPP_LLM_CHECK(suite, batch_stats.completed_requests == 2);
        CPP_LLM_CHECK(suite, batch_stats.model_batch_calls == 4);
        CPP_LLM_CHECK(suite, batch_stats.tensor_batch_steps == 3);
        CPP_LLM_CHECK(suite, batch_a.generated() == sequential_a.generated());
        CPP_LLM_CHECK(suite, batch_b.generated() == sequential_b.generated());
        CPP_LLM_CHECK(suite, batch_a.cache().size() == sequential_a.cache().size());
        CPP_LLM_CHECK(suite, batch_b.cache().size() == sequential_b.cache().size());
        CPP_LLM_CHECK(suite, batch_a.finished() && batch_b.finished());
        const std::vector<std::pair<std::size_t, std::size_t>> expected{
            {10, 0}, {11, 0}, {10, 1}, {11, 1}, {10, 2}, {11, 2},
        };
        CPP_LLM_CHECK(suite, streamed_steps == expected);
    });

    suite.run("batch admission rejects only overlong request", [&] {
        Embedding embedding(64);
        PositionEncoding position_encoding(32);
        Transformer transformer(2, embedding, position_encoding);
        GenerationConfig config;
        config.max_new_tokens = 3;
        config.max_context_tokens = 4;
        Runtime runtime(config, transformer);
        Request accepted({1});
        Request rejected({1, 2});

        const auto result = runtime.prefill_batch({&accepted, &rejected});
        CPP_LLM_CHECK(suite, result.model_calls == 1);
        CPP_LLM_CHECK(suite, result.prefilled_requests == 1);
        CPP_LLM_CHECK(suite, accepted.prefilled());
        CPP_LLM_CHECK(suite, rejected.phase() == RequestPhase::Rejected);
        CPP_LLM_CHECK(suite, rejected.cache().empty());
    });

    return suite.finish();
}
