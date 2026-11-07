// Chapter 10：验证同步 Completion、SSE、断连、背压和 metrics 契约。
// 职责：证明 Transport-neutral Adapter 不泄漏 ServiceCore 请求。
// 验证入口：chapter10_http_contract_test。
#include "chapter10_serving/http_adapter.h"

#include "chapter06_generation/embedding.h"
#include "chapter06_generation/position_encoding.h"
#include "chapter06_generation/runtime.h"
#include "chapter06_generation/sampler.h"
#include "chapter06_generation/transformer.h"
#include "cpp_llm_engine/testing/test.h"

#include <string>
#include <stdexcept>

namespace {

class RecordingWriter : public chapter10_serving::ResponseWriter {
public:
    bool write_headers(const chapter10_serving::ResponseHead& value) override {
        head = value;
        headers_written = true;
        return connected && headers_allowed;
    }
    chapter10_serving::WriteResult write_event(std::string_view event) override {
        if (!connected) return chapter10_serving::WriteResult::client_closed;
        if (blocked) return chapter10_serving::WriteResult::backpressure;
        body += event;
        return chapter10_serving::WriteResult::ok;
    }
    void finish() override { finished = true; }
    bool client_connected() const override { return connected; }

    chapter10_serving::ResponseHead head;
    std::string body;
    bool connected = true;
    bool headers_allowed = true;
    bool blocked = false;
    bool headers_written = false;
    bool finished = false;
};

class ThrowingWriter final : public RecordingWriter {
public:
    chapter10_serving::WriteResult write_event(std::string_view) override {
        throw std::runtime_error("writer failure");
    }
};

}  // namespace

int main() {
    using namespace chapter06_generation;
    using namespace chapter10_serving;
    cpp_llm_engine::testing::TestSuite suite("chapter10_http_contract");

    GenerationConfig config;
    config.max_new_tokens = 2;
    config.max_context_tokens = 32;
    Transformer transformer(2, Embedding(32), PositionEncoding(16));
    Runtime runtime(config, transformer);
    GreedySampler sampler;
    ServiceCore service(8, 4);
    HttpAdapter adapter(service, runtime, sampler);

    suite.run("health sync completion and metrics share one service", [&] {
        CPP_LLM_CHECK_EQ(suite, adapter.handle({"GET", "/health", {}}).status_code, 200);
        const auto completion = adapter.handle({
            "POST", "/v1/completions",
            "{\"request_id\":1,\"prompt_tokens\":[1,2]}"});
        CPP_LLM_CHECK_EQ(suite, completion.status_code, 200);
        CPP_LLM_CHECK(
            suite, completion.body.find("\"finish_reason\":\"stop\"") != std::string::npos);
        CPP_LLM_CHECK_EQ(suite, service.inflight_size(), 0U);
        const auto metrics = adapter.handle({"GET", "/metrics", {}});
        CPP_LLM_CHECK_EQ(suite, metrics.status_code, 200);
        CPP_LLM_CHECK(
            suite, metrics.body.find("serving_completed_total 1") != std::string::npos);
        CPP_LLM_CHECK(
            suite, metrics.body.find("serving_released_total 1") != std::string::npos);
    });

    suite.run("SSE preserves token order and emits one finish event", [&] {
        RecordingWriter writer;
        const auto response = adapter.handle_stream({
            "POST", "/v1/completions",
            "{\"request_id\":2,\"prompt_tokens\":[3]}"}, writer);
        CPP_LLM_CHECK_EQ(suite, response.status_code, 200);
        CPP_LLM_CHECK(suite, writer.headers_written && writer.finished);
        CPP_LLM_CHECK(suite, writer.head.content_type == "text/event-stream");
        CPP_LLM_CHECK(suite, writer.body.find("\"step\":0") != std::string::npos);
        CPP_LLM_CHECK(suite, writer.body.find("\"step\":1") != std::string::npos);
        CPP_LLM_CHECK(
            suite, writer.body.find("finish_reason\":\"stop") != std::string::npos);
        CPP_LLM_CHECK_EQ(suite, service.inflight_size(), 0U);
    });

    suite.run("disconnect and backpressure cancel and release requests", [&] {
        RecordingWriter disconnect;
        HttpRequest disconnect_request{
            "POST", "/v1/completions",
            "{\"request_id\":3,\"prompt_tokens\":[4]}"};
        disconnect_request.disconnect_after_tokens = 1;
        CPP_LLM_CHECK_EQ(
            suite, adapter.handle_stream(disconnect_request, disconnect).status_code, 499);

        RecordingWriter backpressure;
        backpressure.blocked = true;
        CPP_LLM_CHECK_EQ(
            suite,
            adapter.handle_stream({
                "POST", "/v1/completions",
                "{\"request_id\":4,\"prompt_tokens\":[5]}"}, backpressure).status_code,
            503);
        CPP_LLM_CHECK(suite, backpressure.body.empty());
        CPP_LLM_CHECK(suite, backpressure.finished);
        CPP_LLM_CHECK_EQ(suite, service.inflight_size(), 0U);
        CPP_LLM_CHECK_EQ(suite, service.metrics().cancelled_total, 2U);
    });

    suite.run("header backpressure returns 503 and releases request", [&] {
        RecordingWriter writer;
        writer.headers_allowed = false;
        const auto response = adapter.handle_stream({
            "POST", "/v1/completions",
            "{\"request_id\":5,\"prompt_tokens\":[6]}"}, writer);
        CPP_LLM_CHECK_EQ(suite, response.status_code, 503);
        CPP_LLM_CHECK(suite, writer.finished);
        CPP_LLM_CHECK_EQ(suite, service.inflight_size(), 0U);
    });

    suite.run("invalid body and route fail before admission", [&] {
        CPP_LLM_CHECK_EQ(
            suite,
            adapter.handle({"POST", "/v1/completions", "{}"}).status_code,
            400);
        CPP_LLM_CHECK_EQ(
            suite,
            adapter.handle({
                "POST", "/v1/completions",
                "{\"request_id\":999999999999999999999999999999,"
                "\"prompt_tokens\":[1]}"}).status_code,
            400);
        CPP_LLM_CHECK_EQ(
            suite,
            adapter.handle({
                "POST", "/v1/completions",
                "{\"request_id\":6,"
                "\"prompt_tokens\":[999999999999999999999999999999]}"}).status_code,
            400);
        CPP_LLM_CHECK_EQ(
            suite,
            adapter.handle({"GET", "/missing", {}}).status_code,
            404);
    });

    suite.run("writer exception fails and releases the request", [&] {
        ThrowingWriter writer;
        const auto response = adapter.handle_stream({
            "POST", "/v1/completions",
            "{\"request_id\":7,\"prompt_tokens\":[7]}"}, writer);
        CPP_LLM_CHECK_EQ(suite, response.status_code, 500);
        CPP_LLM_CHECK(suite, writer.finished);
        CPP_LLM_CHECK_EQ(suite, service.inflight_size(), 0U);
    });

    return suite.finish();
}
