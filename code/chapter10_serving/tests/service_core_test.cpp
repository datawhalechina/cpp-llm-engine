// Chapter 10：验证准入、等待/运行迁移和 Chapter08 step 调度。
// 职责：确认服务容量与调度预算分离，完成后可以统一释放。
// 验证入口：chapter10_service_core_test。
#include "chapter10_serving/service_core.h"

#include "chapter06_generation/embedding.h"
#include "chapter06_generation/position_encoding.h"
#include "chapter06_generation/request.h"
#include "chapter06_generation/runtime.h"
#include "chapter06_generation/sampler.h"
#include "chapter06_generation/transformer.h"
#include "cpp_llm_engine/testing/test.h"

#include <algorithm>
#include <stdexcept>
#include <memory>
#include <string>
#include <vector>

namespace {

class BatchTestExecutor final : public chapter06_generation::RequestModelExecutor {
public:
    explicit BatchTestExecutor(chapter06_generation::TokenId token) : token_(token) {}

    bool prefill(
        const chapter06_generation::TokenSequence& prompt,
        chapter06_generation::Logits* logits,
        std::string*) override {
        position_ = prompt.size();
        logits->assign(8, 0.0f);
        (*logits)[token_] = 1.0f;
        return true;
    }

    bool decode(
        chapter06_generation::TokenId,
        chapter06_generation::Logits* logits,
        std::string*) override {
        ++position_;
        logits->assign(8, 0.0f);
        (*logits)[token_] = 1.0f;
        return true;
    }

    chapter06_generation::ModelExecutionState execution_state() const override {
        return {true, position_, 1, true};
    }

private:
    chapter06_generation::TokenId token_ = 0;
    std::size_t position_ = 0;
};

class RecordingBatchRunner final : public chapter06_generation::ExecutorBatchRunner {
public:
    bool decode_batch(
        const std::vector<std::shared_ptr<chapter06_generation::RequestModelExecutor>>&
            executors,
        const chapter06_generation::TokenSequence& tokens,
        const std::vector<std::size_t>& positions,
        const std::vector<std::size_t>& request_ids,
        std::vector<chapter06_generation::Logits>* logits,
        std::string* error) const override {
        ++calls;
        max_batch_size = std::max(max_batch_size, executors.size());
        if (executors.size() != tokens.size() || executors.size() != positions.size() ||
            executors.size() != request_ids.size()) {
            if (error != nullptr) *error = "batch test input size mismatch";
            return false;
        }
        logits->resize(executors.size());
        for (std::size_t index = 0; index < executors.size(); ++index) {
            if (executors[index]->execution_state().committed_tokens != positions[index] ||
                !executors[index]->decode(tokens[index], &(*logits)[index], error)) {
                return false;
            }
        }
        return true;
    }

    mutable std::size_t calls = 0;
    mutable std::size_t max_batch_size = 0;
};

}  // namespace

int main() {
    using namespace chapter06_generation;
    using namespace chapter10_serving;
    cpp_llm_engine::testing::TestSuite suite("chapter10_service_core");

    suite.run("capacity and identity are admission concerns", [&] {
        GreedySampler sampler;
        Request first({1});
        Request second({2});
        Request overflow({3});
        ServiceCore service(2, 1);
        CPP_LLM_CHECK(suite, service.admit(1, first, sampler).accepted());
        CPP_LLM_CHECK(suite, service.admit(2, second, sampler).accepted());
        CPP_LLM_CHECK(
            suite,
            service.admit(1, overflow, sampler).status == AdmissionStatus::duplicate_id);
        CPP_LLM_CHECK(
            suite,
            service.admit(3, overflow, sampler).status == AdmissionStatus::queue_full);
        CPP_LLM_CHECK_EQ(suite, service.waiting_size(), 2U);
        CPP_LLM_CHECK_EQ(suite, service.metrics().peak_inflight, 2U);
        CPP_LLM_CHECK_THROWS(suite, std::invalid_argument, ServiceCore(1, 2));
    });

    suite.run("service submits running requests to chapter08 one step at a time", [&] {
        GenerationConfig config;
        config.max_new_tokens = 2;
        config.max_context_tokens = 32;
        Transformer transformer(2, Embedding(32), PositionEncoding(16));
        Runtime runtime(config, transformer);
        GreedySampler sampler;
        Request first({1});
        Request second({2});
        ServiceCore service(2, 1);
        CPP_LLM_REQUIRE(suite, service.admit(10, first, sampler).accepted());
        CPP_LLM_REQUIRE(suite, service.admit(11, second, sampler).accepted());

        std::size_t callbacks = 0;
        while (!first.terminal()) {
            const auto step = service.dispatch_step(
                runtime, 2,
                [&](std::size_t request_id, TokenId, std::size_t) {
                    CPP_LLM_CHECK_EQ(suite, request_id, 10U);
                    ++callbacks;
                });
            CPP_LLM_CHECK_EQ(suite, step.scheduled_requests, 1U);
        }
        CPP_LLM_CHECK_EQ(suite, callbacks, 2U);
        CPP_LLM_CHECK_EQ(suite, service.waiting_size(), 1U);
        CPP_LLM_CHECK_EQ(suite, service.release_terminal(), 1U);

        while (!second.terminal()) service.dispatch_step(runtime, 2);
        CPP_LLM_CHECK_EQ(suite, service.release_terminal(), 1U);
        CPP_LLM_CHECK_EQ(suite, service.inflight_size(), 0U);
        CPP_LLM_CHECK_EQ(suite, service.metrics().completed_total, 2U);
        CPP_LLM_CHECK_EQ(suite, service.metrics().generated_tokens_total, 4U);
    });

    suite.run("admission priority reaches the chapter08 planner", [&] {
        GenerationConfig config;
        config.max_new_tokens = 1;
        Runtime runtime(config);
        GreedySampler sampler;
        Request low({1});
        Request high({2});
        ServiceCore service(2, 1);
        CPP_LLM_REQUIRE(
            suite,
            service.admit(
                20, low, sampler, std::make_shared<BatchTestExecutor>(3), 0)
                .accepted());
        CPP_LLM_REQUIRE(
            suite,
            service.admit(
                21, high, sampler, std::make_shared<BatchTestExecutor>(4), 2)
                .accepted());

        std::size_t selected_id = 0;
        const auto step = service.dispatch_step(
            runtime,
            1,
            [&](std::size_t request_id, TokenId, std::size_t) {
                selected_id = request_id;
            });
        CPP_LLM_CHECK_EQ(suite, step.scheduled_requests, 1U);
        CPP_LLM_CHECK_EQ(suite, selected_id, 21U);
        CPP_LLM_CHECK(suite, high.finished());
        CPP_LLM_CHECK(suite, !low.prefilled());
    });

    suite.run("service dispatches decoder candidates through one batch call", [&] {
        GenerationConfig config;
        config.max_new_tokens = 2;
        Runtime runtime(config);
        GreedySampler sampler;
        Request first({1});
        Request second({2});
        RecordingBatchRunner batch_runner;
        ServiceCore service(2, 2, {}, {}, {}, &batch_runner);
        CPP_LLM_REQUIRE(
            suite,
            service.admit(
                30, first, sampler, std::make_shared<BatchTestExecutor>(3))
                .accepted());
        CPP_LLM_REQUIRE(
            suite,
            service.admit(
                31, second, sampler, std::make_shared<BatchTestExecutor>(4))
                .accepted());

        const auto prefill = service.dispatch_step(runtime, 2);
        CPP_LLM_CHECK_EQ(suite, prefill.generated_tokens, 2U);
        CPP_LLM_CHECK_EQ(suite, prefill.model_batch_calls, 0U);
        const auto decode = service.dispatch_step(runtime, 2);
        CPP_LLM_CHECK_EQ(suite, decode.model_batch_calls, 1U);
        CPP_LLM_CHECK_EQ(suite, decode.tensor_batch_requests, 2U);
        CPP_LLM_CHECK_EQ(suite, batch_runner.calls, 1U);
        CPP_LLM_CHECK_EQ(suite, batch_runner.max_batch_size, 2U);
        CPP_LLM_CHECK(suite, first.finished() && second.finished());
        CPP_LLM_CHECK(suite, first.generated() == TokenSequence({3, 3}));
        CPP_LLM_CHECK(suite, second.generated() == TokenSequence({4, 4}));
    });

    return suite.finish();
}
