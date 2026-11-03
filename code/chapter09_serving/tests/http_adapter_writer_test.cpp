// Chapter 09：验证 HttpAdapter 通过 ResponseWriter 输出真实 token event。
// 职责：覆盖正常 stream、disconnect、timeout 和背压，不依赖具体 socket 或网络库。
// 验证入口：Chapter 09 CTest target 或 scripts/run.sh。
#include "chapter06_generation/embedding.h"
#include "chapter06_generation/position_encoding.h"
#include "chapter06_generation/runtime.h"
#include "chapter06_generation/sampler.h"
#include "chapter06_generation/transformer.h"
#include "chapter09_serving/http_adapter.h"

#include "cpp_llm_engine/testing/test.h"

#include <string>

namespace {

class RecordingWriter final : public chapter09_serving::ResponseWriter {
public:
    bool write_headers(const chapter09_serving::ResponseHead& head) override {
        headers = head;
        headers_written = true;
        return connected;
    }

    chapter09_serving::WriteResult write_event(std::string_view event) override {
        if (!connected) return chapter09_serving::WriteResult::ClientClosed;
        if (blocked) return chapter09_serving::WriteResult::Backpressure;
        body += event;
        return chapter09_serving::WriteResult::Ok;
    }

    void finish() override { finished = true; }
    bool client_connected() const override { return connected; }

    chapter09_serving::ResponseHead headers;
    std::string body;
    bool connected = true;
    bool blocked = false;
    bool headers_written = false;
    bool finished = false;
};

}  // namespace

int main() {
    using namespace chapter06_generation;
    using namespace chapter09_serving;

    GenerationConfig config;
    config.max_new_tokens = 2;
    config.max_context_tokens = 32;
    Embedding embedding(32);
    PositionEncoding positions(16);
    Transformer transformer(2, embedding, positions);
    Runtime runtime(config, transformer);
    GreedySampler sampler;
    ServingCore service(4, 4);
    HttpAdapter adapter(service, runtime, sampler);
    cpp_llm_engine::testing::TestSuite suite("chapter09_http_adapter_writer");

    suite.run("normal stream writes headers tokens and finish event", [&] {
        RecordingWriter normal;
        const HttpResponse response = adapter.handle_stream({
            "POST", "/v1/completions",
            "{\"request_id\":11,\"prompt_tokens\":[1,2]}"}, normal);
        CPP_LLM_CHECK_EQ(suite, response.status_code, 200);
        CPP_LLM_CHECK(suite, normal.headers_written);
        CPP_LLM_CHECK_EQ(suite, normal.headers.status_code, 200);
        CPP_LLM_CHECK(suite, normal.headers.content_type == "text/event-stream");
        CPP_LLM_CHECK(suite, normal.body.find("data: {\"token\":") != std::string::npos);
        CPP_LLM_CHECK(suite, normal.body.find("finish_reason\":\"stop") != std::string::npos);
        CPP_LLM_CHECK(suite, normal.finished);
    });

    suite.run("disconnect and timeout map to terminal stream events", [&] {
        RecordingWriter disconnected;
        HttpRequest disconnect_request{
            "POST", "/v1/completions",
            "{\"request_id\":12,\"prompt_tokens\":[1,2]}"};
        disconnect_request.disconnect_after_tokens = 1;
        const HttpResponse disconnect_response =
            adapter.handle_stream(disconnect_request, disconnected);
        CPP_LLM_CHECK_EQ(suite, disconnect_response.status_code, 499);
        CPP_LLM_CHECK(
            suite,
            disconnected.body.find("finish_reason\":\"client_disconnect") !=
                std::string::npos);

        RecordingWriter timed_out;
        HttpRequest timeout_request{
            "POST", "/v1/completions",
            "{\"request_id\":13,\"prompt_tokens\":[1,2]}"};
        timeout_request.timeout_after_tokens = 1;
        const HttpResponse timeout_response =
            adapter.handle_stream(timeout_request, timed_out);
        CPP_LLM_CHECK_EQ(suite, timeout_response.status_code, 408);
        CPP_LLM_CHECK(
            suite,
            timed_out.body.find("finish_reason\":\"timeout") != std::string::npos);
    });

    suite.run("backpressure stops stream without buffering an event", [&] {
        RecordingWriter writer;
        writer.blocked = true;
        const HttpResponse response = adapter.handle_stream({
            "POST", "/v1/completions",
            "{\"request_id\":14,\"prompt_tokens\":[1,2]}"}, writer);
        CPP_LLM_CHECK_EQ(suite, response.status_code, 503);
        CPP_LLM_CHECK(suite, writer.body.empty());
        CPP_LLM_CHECK(suite, writer.finished);
    });
    return suite.finish();
}
