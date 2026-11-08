// Chapter 08：验证 Ragged Tensor Batch 的输入、提交和逐请求等价性。
// 职责：固定批量 Prefill/Decode 契约，不测试调度预算或网络并发。
// 验证入口：chapter08_tensor_batch_test。
#include "chapter08_scheduling/scheduler.h"

#include "chapter06_generation/embedding.h"
#include "chapter06_generation/position_encoding.h"
#include "chapter06_generation/request.h"
#include "chapter06_generation/runtime.h"
#include "chapter06_generation/sampler.h"
#include "chapter06_generation/transformer.h"
#include "cpp_llm_engine/testing/test.h"

#include <cstddef>
#include <utility>
#include <vector>

int main() {
    using namespace chapter06_generation;
    using namespace chapter08_scheduling;
    cpp_llm_engine::testing::TestSuite suite("chapter08_tensor_batch");

    suite.run("embedding batch exposes valid ragged offsets", [&] {
        Embedding embedding(64);
        const RaggedBatch embedded = embedding.forward_batch({{1, 2, 3}, {4, 5}});
        CPP_LLM_CHECK(suite, embedded.valid());
        CPP_LLM_CHECK(suite, embedded.batch_size() == 2);
        CPP_LLM_CHECK(
            suite, embedded.row_offsets == std::vector<std::size_t>({0, 3, 5}));
    });

    suite.run("batched prefill matches individual calls", [&] {
        Embedding embedding(64);
        PositionEncoding positions(32);
        Transformer transformer(2, embedding, positions);
        Request request_a({1, 2, 3});
        Request request_b({4, 5});
        const auto batch = transformer.prefill_batch({&request_a, &request_b});
        CPP_LLM_REQUIRE(suite, batch.size() == 2);
        CPP_LLM_CHECK(suite, batch[0] == transformer.prefill(request_a));
        CPP_LLM_CHECK(suite, batch[1] == transformer.prefill(request_b));
    });

    suite.run("duplicate request is rejected without partial mutation", [&] {
        Embedding embedding(64);
        PositionEncoding positions(32);
        Transformer transformer(2, embedding, positions);
        GenerationConfig config;
        config.max_new_tokens = 2;
        Runtime runtime(config, transformer);
        Request request({7, 8});

        const auto result = runtime.prefill_batch({&request, &request});
        CPP_LLM_CHECK(suite, result.model_calls == 0);
        CPP_LLM_CHECK(suite, !request.prefilled());
        CPP_LLM_CHECK(suite, request.history().empty());
    });

    suite.run("batch admission rejects only the overlong request", [&] {
        Embedding embedding(64);
        PositionEncoding positions(32);
        Transformer transformer(2, embedding, positions);
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
        CPP_LLM_CHECK(suite, rejected.history().empty());
    });

    return suite.finish();
}
