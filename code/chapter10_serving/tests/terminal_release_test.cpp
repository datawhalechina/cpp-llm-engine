// Chapter 10：验证 waiting/running 的取消、完成和 K/V 释放使用同一路径。
// 职责：锁定幂等 release 与 Chapter07 BlockPool 容量恢复。
// 验证入口：chapter10_terminal_release_test。
#include "chapter10_serving/service_core.h"

#include "chapter06_generation/embedding.h"
#include "chapter06_generation/position_encoding.h"
#include "chapter06_generation/request.h"
#include "chapter06_generation/runtime.h"
#include "chapter06_generation/sampler.h"
#include "chapter06_generation/transformer.h"
#include "chapter07_kv_cache/block_table.h"
#include "cpp_llm_engine/testing/test.h"

#include <memory>

int main() {
    using namespace chapter06_generation;
    using namespace chapter10_serving;
    cpp_llm_engine::testing::TestSuite suite("chapter10_terminal_release");

    suite.run("waiting cancellation and running completion release together", [&] {
        GenerationConfig config;
        config.max_new_tokens = 1;
        config.max_context_tokens = 16;
        Transformer transformer(2, Embedding(32), PositionEncoding(16));
        Runtime runtime(config, transformer);
        GreedySampler sampler;
        auto pool = std::make_shared<chapter07_kv_cache::KvBlockPool>(2, 2);
        ServiceCore service(2, 1, ServiceNow{}, pool);
        Request running({1});
        Request waiting({2});
        CPP_LLM_REQUIRE(suite, service.admit(20, running, sampler).accepted());
        CPP_LLM_REQUIRE(suite, service.admit(21, waiting, sampler).accepted());
        CPP_LLM_REQUIRE(suite, pool->reserve(21, 1));
        CPP_LLM_REQUIRE(suite, pool->commit_tokens(21, 1));
        CPP_LLM_CHECK(suite, service.cancel(21));

        const auto step = service.dispatch_step(runtime, 2);
        CPP_LLM_CHECK_EQ(suite, step.generated_tokens, 1U);
        CPP_LLM_CHECK(suite, running.finished());
        CPP_LLM_CHECK(suite, waiting.cancelled());
        CPP_LLM_CHECK_EQ(suite, service.release_terminal(), 2U);
        CPP_LLM_CHECK_EQ(suite, service.release_terminal(), 0U);
        CPP_LLM_CHECK_EQ(suite, pool->free_blocks(), pool->total_blocks());
        CPP_LLM_CHECK_EQ(suite, service.inflight_size(), 0U);
        CPP_LLM_CHECK_EQ(suite, service.metrics().completed_total, 1U);
        CPP_LLM_CHECK_EQ(suite, service.metrics().cancelled_total, 1U);
        CPP_LLM_CHECK_EQ(suite, service.metrics().released_total, 2U);
    });

    suite.run("timeout remains owned until unified release", [&] {
        GreedySampler sampler;
        Request request({3});
        ServiceCore service(1, 1);
        CPP_LLM_REQUIRE(suite, service.admit(30, request, sampler).accepted());
        CPP_LLM_CHECK(suite, service.timeout(30));
        CPP_LLM_CHECK_EQ(suite, service.waiting_size(), 1U);
        CPP_LLM_CHECK_EQ(suite, service.release_terminal(), 1U);
        CPP_LLM_CHECK_EQ(suite, service.metrics().timed_out_total, 1U);
    });

    return suite.finish();
}
