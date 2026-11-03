// Engine 集成测试：验证预计算 embedding 复用 Chapter09 同步与 SSE 服务链路。
// 覆盖：每请求 Executor、token 一致性、失败终态、metrics 和 Context 释放。
// 依赖：使用合成 Decoder Context，不依赖模型、图片、socket、Python 或 GPU。
// 运行：由 Engine CTest 执行 embedding_request_service_test。
#include "cpp_llm_engine/embedding_request_executor.h"

#include "chapter06_generation/runtime.h"
#include "chapter06_generation/sampler.h"
#include "chapter09_serving/http_adapter.h"
#include "cpp_llm_engine/testing/test.h"

#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

struct ContextState {
    std::size_t embedding_prefill_calls = 0;
    std::size_t decode_calls = 0;
};

class ServiceEmbeddingContext final
    : public cpp_llm_engine::core::DecoderExecutionContext {
public:
    ServiceEmbeddingContext(std::shared_ptr<ContextState> state, bool fail_prefill)
        : state_(std::move(state)), fail_prefill_(fail_prefill) {}

    void reset() override { token_count_ = 0; }

    cpp_llm_engine::core::Status prefill_tokens(
        const cpp_llm_engine::core::DecoderTokenSequence&,
        cpp_llm_engine::core::DecoderLogits*) override {
        return cpp_llm_engine::core::Status::error("token prefill is not expected");
    }

    cpp_llm_engine::core::Status prefill_embeddings(
        const cpp_llm_engine::core::DecoderEmbeddingSequence& embeddings,
        cpp_llm_engine::core::DecoderLogits* logits) override {
        ++state_->embedding_prefill_calls;
        if (fail_prefill_) {
            return cpp_llm_engine::core::Status::error("synthetic visual prefill failure");
        }
        token_count_ = embeddings.token_count;
        logits->assign(5, 0.0f);
        (*logits)[2] = 1.0f;
        return cpp_llm_engine::core::Status::ok();
    }

    cpp_llm_engine::core::Status decode_token(
        cpp_llm_engine::core::DecoderTokenId,
        cpp_llm_engine::core::DecoderLogits* logits) override {
        ++state_->decode_calls;
        ++token_count_;
        logits->assign(5, 0.0f);
        (*logits)[4] = 1.0f;
        return cpp_llm_engine::core::Status::ok();
    }

    std::size_t token_count() const override { return token_count_; }

    cpp_llm_engine::core::DecoderModelContract model_contract() const override {
        cpp_llm_engine::core::DecoderLayerContract layer;
        layer.hidden_size = 4;
        layer.ffn_hidden_size = 8;
        layer.query_head_count = 1;
        layer.kv_head_count = 1;
        layer.head_dimension = 4;
        return {5, 2, 16, layer};
    }

private:
    std::shared_ptr<ContextState> state_;
    bool fail_prefill_ = false;
    std::size_t token_count_ = 0;
};

class RecordingWriter final : public chapter09_serving::ResponseWriter {
public:
    bool write_headers(const chapter09_serving::ResponseHead& value) override {
        head = value;
        return true;
    }

    chapter09_serving::WriteResult write_event(std::string_view event) override {
        body += event;
        return chapter09_serving::WriteResult::Ok;
    }

    void finish() override { finished = true; }
    bool client_connected() const override { return true; }

    chapter09_serving::ResponseHead head;
    std::string body;
    bool finished = false;
};

cpp_llm_engine::core::DecoderEmbeddingSequence make_embeddings() {
    return {3, 4, {1.0f, 2.0f, 3.0f, 4.0f,
                   5.0f, 6.0f, 7.0f, 8.0f,
                   9.0f, 10.0f, 11.0f, 12.0f}};
}

std::shared_ptr<chapter06_generation::RequestModelExecutor> make_executor(
    bool fail_prefill,
    std::vector<std::weak_ptr<ContextState>>* observed_states) {
    auto state = std::make_shared<ContextState>();
    observed_states->push_back(state);
    auto context = std::make_unique<ServiceEmbeddingContext>(state, fail_prefill);
    std::string error;
    auto executor = cpp_llm_engine::EmbeddingRequestExecutor::create(
        std::move(context), make_embeddings(), &error);
    if (executor == nullptr || !error.empty()) {
        throw std::runtime_error("failed to create embedding request executor: " + error);
    }
    return executor;
}

}  // namespace

int main() {
    cpp_llm_engine::testing::TestSuite suite("embedding_request_service");
    chapter06_generation::GenerationConfig config;
    config.max_new_tokens = 2;
    config.max_context_tokens = 16;
    config.eos_token_id = 4;
    chapter06_generation::Runtime runtime(config);
    chapter06_generation::GreedySampler sampler;
    suite.run("sync and SSE requests release per-request contexts", [&] {
        chapter09_serving::ServingCore service(2, 2);
        std::vector<std::weak_ptr<ContextState>> states;
        chapter09_serving::HttpAdapter adapter(
            service,
            runtime,
            sampler,
            [&states]() { return make_executor(false, &states); });

        const auto sync = adapter.handle({
            "POST", "/v1/completions",
            "{\"request_id\":71,\"prompt_tokens\":[0,0,0]}"});
        CPP_LLM_CHECK_EQ(suite, sync.status_code, 200);
        CPP_LLM_CHECK(
            suite, sync.body.find("\"generated_token_ids\":[2,4]") != std::string::npos);
        CPP_LLM_REQUIRE(suite, states.size() == 1);
        CPP_LLM_CHECK(suite, states[0].expired());

        RecordingWriter writer;
        const auto stream = adapter.handle_stream({
            "POST", "/v1/completions",
            "{\"request_id\":72,\"prompt_tokens\":[0,0,0]}"}, writer);
        CPP_LLM_CHECK_EQ(suite, stream.status_code, 200);
        CPP_LLM_CHECK_EQ(suite, writer.head.status_code, 200);
        CPP_LLM_CHECK(suite, writer.head.content_type == "text/event-stream");
        CPP_LLM_CHECK(
            suite, writer.body.find("data: {\"token\":2,\"step\":0}") != std::string::npos);
        CPP_LLM_CHECK(
            suite, writer.body.find("data: {\"token\":4,\"step\":1}") != std::string::npos);
        CPP_LLM_CHECK(suite, writer.body.find("finish_reason\":\"stop") != std::string::npos);
        CPP_LLM_CHECK(suite, writer.finished);
        CPP_LLM_REQUIRE(suite, states.size() == 2);
        CPP_LLM_CHECK(suite, states[1].expired());
        CPP_LLM_CHECK_EQ(suite, service.metrics().accepted_total, 2U);
        CPP_LLM_CHECK_EQ(suite, service.metrics().completed_total, 2U);
        CPP_LLM_CHECK_EQ(suite, service.metrics().released_total, 2U);
        CPP_LLM_CHECK_EQ(suite, service.metrics().generated_tokens_total, 4U);
    });

    suite.run("failed visual prefill reaches terminal state and releases context", [&] {
        chapter09_serving::ServingCore service(1, 1);
        std::vector<std::weak_ptr<ContextState>> states;
        chapter09_serving::HttpAdapter adapter(
            service,
            runtime,
            sampler,
            [&states]() { return make_executor(true, &states); });
        const auto response = adapter.handle({
            "POST", "/v1/completions",
            "{\"request_id\":73,\"prompt_tokens\":[0,0,0]}"});
        CPP_LLM_CHECK_EQ(suite, response.status_code, 500);
        CPP_LLM_CHECK(
            suite, response.body.find("runtime_did_not_finish") != std::string::npos);
        CPP_LLM_CHECK_EQ(suite, service.metrics().failed_total, 1U);
        CPP_LLM_CHECK_EQ(suite, service.metrics().released_total, 1U);
        CPP_LLM_REQUIRE(suite, states.size() == 1);
        CPP_LLM_CHECK(suite, states[0].expired());
    });

    suite.run("mixed sync and SSE requests do not retain contexts", [&] {
        chapter09_serving::ServingCore service(4, 4);
        std::vector<std::weak_ptr<ContextState>> states;
        chapter09_serving::HttpAdapter adapter(
            service,
            runtime,
            sampler,
            [&states]() { return make_executor(false, &states); });
        constexpr std::size_t kRequests = 32;
        for (std::size_t index = 0; index < kRequests; ++index) {
            const std::string body = "{\"request_id\":" +
                std::to_string(1000 + index) + ",\"prompt_tokens\":[0,0,0]}";
            if (index % 2 == 0) {
                const auto response = adapter.handle({"POST", "/v1/completions", body});
                CPP_LLM_CHECK_EQ(suite, response.status_code, 200);
                CPP_LLM_CHECK(
                    suite,
                    response.body.find("\"generated_token_ids\":[2,4]") !=
                        std::string::npos);
            } else {
                RecordingWriter writer;
                const auto response =
                    adapter.handle_stream({"POST", "/v1/completions", body}, writer);
                CPP_LLM_CHECK_EQ(suite, response.status_code, 200);
                CPP_LLM_CHECK(
                    suite,
                    writer.body.find("data: {\"token\":2,\"step\":0}") !=
                        std::string::npos);
                CPP_LLM_CHECK(
                    suite,
                    writer.body.find("data: {\"token\":4,\"step\":1}") !=
                        std::string::npos);
                CPP_LLM_CHECK(suite, writer.finished);
            }
            CPP_LLM_REQUIRE(suite, states.size() == index + 1);
            CPP_LLM_CHECK(suite, states.back().expired());
            CPP_LLM_CHECK_EQ(suite, service.inflight_size(), 0U);
        }
        CPP_LLM_CHECK_EQ(suite, service.metrics().accepted_total, kRequests);
        CPP_LLM_CHECK_EQ(suite, service.metrics().completed_total, kRequests);
        CPP_LLM_CHECK_EQ(suite, service.metrics().released_total, kRequests);
        CPP_LLM_CHECK_EQ(suite, service.metrics().generated_tokens_total, kRequests * 2);
    });

    suite.run("repeated failures release every context", [&] {
        chapter09_serving::ServingCore service(2, 2);
        std::vector<std::weak_ptr<ContextState>> states;
        chapter09_serving::HttpAdapter adapter(
            service,
            runtime,
            sampler,
            [&states]() { return make_executor(true, &states); });
        constexpr std::size_t kRequests = 8;
        for (std::size_t index = 0; index < kRequests; ++index) {
            const std::string body = "{\"request_id\":" +
                std::to_string(2000 + index) + ",\"prompt_tokens\":[0,0,0]}";
            const auto response = adapter.handle({"POST", "/v1/completions", body});
            CPP_LLM_CHECK_EQ(suite, response.status_code, 500);
            CPP_LLM_REQUIRE(suite, states.size() == index + 1);
            CPP_LLM_CHECK(suite, states.back().expired());
            CPP_LLM_CHECK_EQ(suite, service.inflight_size(), 0U);
        }
        CPP_LLM_CHECK_EQ(suite, service.metrics().accepted_total, kRequests);
        CPP_LLM_CHECK_EQ(suite, service.metrics().failed_total, kRequests);
        CPP_LLM_CHECK_EQ(suite, service.metrics().released_total, kRequests);
    });

    suite.run("disconnect and timeout release terminal contexts", [&] {
        chapter09_serving::ServingCore service(2, 2);
        std::vector<std::weak_ptr<ContextState>> states;
        chapter09_serving::HttpAdapter adapter(
            service,
            runtime,
            sampler,
            [&states]() { return make_executor(false, &states); });
        chapter09_serving::HttpRequest disconnected{
            "POST", "/v1/completions",
            "{\"request_id\":3001,\"prompt_tokens\":[0,0,0]}"};
        disconnected.disconnect_after_tokens = 1;
        CPP_LLM_CHECK_EQ(suite, adapter.handle(disconnected).status_code, 499);
        CPP_LLM_REQUIRE(suite, states.size() == 1);
        CPP_LLM_CHECK(suite, states.back().expired());

        chapter09_serving::HttpRequest timed_out{
            "POST", "/v1/completions",
            "{\"request_id\":3002,\"prompt_tokens\":[0,0,0]}"};
        timed_out.timeout_after_tokens = 1;
        CPP_LLM_CHECK_EQ(suite, adapter.handle(timed_out).status_code, 408);
        CPP_LLM_REQUIRE(suite, states.size() == 2);
        CPP_LLM_CHECK(suite, states.back().expired());
        CPP_LLM_CHECK_EQ(suite, service.metrics().cancelled_total, 1U);
        CPP_LLM_CHECK_EQ(suite, service.metrics().timed_out_total, 1U);
        CPP_LLM_CHECK_EQ(suite, service.metrics().released_total, 2U);
        CPP_LLM_CHECK_EQ(suite, service.inflight_size(), 0U);
    });
    return suite.finish();
}
