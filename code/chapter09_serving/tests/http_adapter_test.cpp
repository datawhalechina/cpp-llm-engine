// Chapter 09：测试 transport-neutral HTTP adapter 的 endpoint 和错误映射。
// 职责：验证 HTTP body 不绕过 ServiceCore，metrics 能观察请求结果。
// 验证入口：Chapter 09 CTest target 或 scripts/run.sh。
#include "chapter06_generation/embedding.h"
#include "chapter06_generation/model_executor.h"
#include "chapter06_generation/position_encoding.h"
#include "chapter06_generation/runtime.h"
#include "chapter06_generation/sampler.h"
#include "chapter06_generation/transformer.h"
#include "chapter09_serving/http_adapter.h"

#include "cpp_llm_engine/testing/test.h"

#include <memory>
#include <string>

namespace {

class FixedHttpExecutor final : public chapter06_generation::RequestModelExecutor {
public:
    bool prefill(const chapter06_generation::TokenSequence&,
                 chapter06_generation::Logits* logits,
                 std::string*) override {
        *logits = fixed_logits();
        return true;
    }

    bool decode(chapter06_generation::TokenId,
                chapter06_generation::Logits* logits,
                std::string*) override {
        *logits = fixed_logits();
        return true;
    }

private:
    static chapter06_generation::Logits fixed_logits() {
        chapter06_generation::Logits logits(8, 0.0f);
        logits[3] = 1.0f;
        return logits;
    }
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
    ServingCore service(2, 2);
    HttpAdapter adapter(service, runtime, sampler);
    cpp_llm_engine::testing::TestSuite suite("chapter09_http_adapter");

    suite.run("health invalid and missing routes map status codes", [&] {
        const HttpResponse health = adapter.handle({"GET", "/health", ""});
        CPP_LLM_CHECK_EQ(suite, health.status_code, 200);
        CPP_LLM_CHECK(suite, health.body == "{\"status\":\"ok\"}");

        const HttpResponse invalid = adapter.handle({"POST", "/v1/completions", "{}"});
        CPP_LLM_CHECK_EQ(suite, invalid.status_code, 400);
        const HttpResponse missing = adapter.handle({"GET", "/missing", ""});
        CPP_LLM_CHECK_EQ(suite, missing.status_code, 404);
    });

    suite.run("completion stream and terminal controls update service metrics", [&] {
        const HttpResponse completion = adapter.handle({
            "POST", "/v1/completions",
            "{\"request_id\":7,\"prompt_tokens\":[1,2,3]}"});
        CPP_LLM_CHECK_EQ(suite, completion.status_code, 200);
        CPP_LLM_CHECK(
            suite,
            completion.body.find("\"finish_reason\":\"stop\"") != std::string::npos);

        const HttpResponse stream = adapter.handle({
            "POST", "/v1/completions",
            "{\"request_id\":8,\"prompt_tokens\":[1,2],\"stream\":true}"});
        CPP_LLM_CHECK_EQ(suite, stream.status_code, 200);
        CPP_LLM_CHECK(suite, stream.content_type == "text/event-stream");
        CPP_LLM_CHECK(suite, stream.body.find("data: {\"token\":") != std::string::npos);
        CPP_LLM_CHECK(suite, stream.body.find("finish_reason\":\"stop") != std::string::npos);

        HttpRequest disconnected{
            "POST", "/v1/completions",
            "{\"request_id\":9,\"prompt_tokens\":[1,2],\"stream\":true}"};
        disconnected.disconnect_after_tokens = 1;
        const HttpResponse disconnect_response = adapter.handle(disconnected);
        CPP_LLM_CHECK_EQ(suite, disconnect_response.status_code, 499);
        CPP_LLM_CHECK(
            suite, disconnect_response.body.find("client_disconnect") != std::string::npos);

        HttpRequest timed_out{
            "POST", "/v1/completions",
            "{\"request_id\":10,\"prompt_tokens\":[1,2],\"stream\":true}"};
        timed_out.timeout_after_tokens = 1;
        const HttpResponse timeout_response = adapter.handle(timed_out);
        CPP_LLM_CHECK_EQ(suite, timeout_response.status_code, 408);
        CPP_LLM_CHECK(
            suite,
            timeout_response.body.find("finish_reason\":\"timeout") != std::string::npos);

        const HttpResponse metrics = adapter.handle({"GET", "/metrics", ""});
        CPP_LLM_CHECK_EQ(suite, metrics.status_code, 200);
        CPP_LLM_CHECK(suite, metrics.content_type.find("text/plain") != std::string::npos);
        CPP_LLM_CHECK(suite, metrics.body.find("serving_accepted_total 4") != std::string::npos);
        CPP_LLM_CHECK(suite, metrics.body.find("serving_completed_total 2") != std::string::npos);
        CPP_LLM_CHECK(suite, metrics.body.find("serving_failed_total 0") != std::string::npos);
        CPP_LLM_CHECK(suite, metrics.body.find("serving_released_total 4") != std::string::npos);
        CPP_LLM_CHECK(
            suite,
            metrics.body.find("# TYPE serving_queue_wait_seconds summary") !=
                std::string::npos);
        CPP_LLM_CHECK(
            suite,
            metrics.body.find("serving_queue_wait_seconds_count 4") != std::string::npos);
        CPP_LLM_CHECK(
            suite, metrics.body.find("serving_ttft_seconds_count 4") != std::string::npos);
        CPP_LLM_CHECK(
            suite,
            metrics.body.find("serving_completion_seconds_count 4") != std::string::npos);
    });

    Runtime injected_runtime(config);
    ServingCore injected_service(2, 2);
    std::size_t factory_calls = 0;
    HttpAdapter injected_adapter(
        injected_service,
        injected_runtime,
        sampler,
        [&factory_calls]() {
            ++factory_calls;
            return std::make_shared<FixedHttpExecutor>();
        });
    suite.run("executor factory controls model path and unavailable error", [&] {
        const HttpResponse injected = injected_adapter.handle({
            "POST", "/v1/completions",
            "{\"request_id\":20,\"prompt_tokens\":[1,2]}"});
        CPP_LLM_CHECK_EQ(suite, injected.status_code, 200);
        CPP_LLM_CHECK(
            suite,
            injected.body.find("\"generated_token_ids\":[3,3]") != std::string::npos);
        CPP_LLM_CHECK_EQ(suite, factory_calls, 1U);

        HttpAdapter unavailable_adapter(
            injected_service,
            injected_runtime,
            sampler,
            []() -> std::shared_ptr<RequestModelExecutor> { return {}; });
        const HttpResponse unavailable = unavailable_adapter.handle({
            "POST", "/v1/completions",
            "{\"request_id\":21,\"prompt_tokens\":[1]}"});
        CPP_LLM_CHECK_EQ(suite, unavailable.status_code, 500);
        CPP_LLM_CHECK(
            suite,
            unavailable.body.find("model_executor_unavailable") != std::string::npos);
    });
    return suite.finish();
}
