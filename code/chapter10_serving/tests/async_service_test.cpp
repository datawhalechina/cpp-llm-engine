// Chapter 10: verify asynchronous admission, event delivery and terminal cleanup.
#include "chapter10_serving/async_service.h"
#include "chapter10_serving/http_adapter.h"

#include "chapter06_generation/embedding.h"
#include "chapter06_generation/position_encoding.h"
#include "chapter06_generation/runtime.h"
#include "chapter06_generation/sampler.h"
#include "chapter06_generation/transformer.h"
#include "cpp_llm_engine/testing/test.h"

#include <cstddef>
#include <chrono>
#include <string>
#include <thread>

namespace {

class RecordingWriter final : public chapter10_serving::ResponseWriter {
public:
    bool write_headers(const chapter10_serving::ResponseHead&) override {
        headers = true;
        return true;
    }
    chapter10_serving::WriteResult write_event(std::string_view value) override {
        body.append(value);
        return chapter10_serving::WriteResult::ok;
    }
    void finish() override { finished = true; }
    bool client_connected() const override { return true; }

    std::string body;
    bool headers = false;
    bool finished = false;
};

}  // namespace

int main() {
    using namespace chapter06_generation;
    using namespace chapter10_serving;
    cpp_llm_engine::testing::TestSuite suite("chapter10_async_service");

    suite.run("network-side handles observe independently owned requests", [&] {
        GenerationConfig config;
        config.max_new_tokens = 3;
        config.max_context_tokens = 32;
        Transformer transformer(2, Embedding(32), PositionEncoding(16));
        Runtime runtime(config, transformer);
        GreedySampler sampler;
        ServiceCore service(4, 4);
        AsyncServiceLoop loop(service, runtime, sampler, 8);

        const AsyncRequestHandle first = loop.submit(10, {1, 2});
        const AsyncRequestHandle second = loop.submit(11, {3, 4});
        std::size_t first_cursor = 0;
        std::size_t second_cursor = 0;
        AsyncTokenEvent event;
        std::size_t first_events = 0;
        std::size_t second_events = 0;
        while (first.wait_next(&first_cursor, &event)) ++first_events;
        while (second.wait_next(&second_cursor, &event)) ++second_events;
        const AsyncRequestResult first_result = first.wait();
        const AsyncRequestResult second_result = second.wait();

        CPP_LLM_CHECK(
            suite, first_result.status == AsyncCompletionStatus::completed);
        CPP_LLM_CHECK(
            suite, second_result.status == AsyncCompletionStatus::completed);
        CPP_LLM_CHECK_EQ(suite, first_events, 3U);
        CPP_LLM_CHECK_EQ(suite, second_events, 3U);
        CPP_LLM_CHECK_EQ(suite, first_result.generated.size(), 3U);
        CPP_LLM_CHECK_EQ(suite, second_result.generated.size(), 3U);
        const AsyncServiceSnapshot snapshot = loop.snapshot();
        CPP_LLM_CHECK_EQ(suite, snapshot.waiting, 0U);
        CPP_LLM_CHECK_EQ(suite, snapshot.running, 0U);
        CPP_LLM_CHECK_EQ(suite, snapshot.metrics.completed_total, 2U);
        CPP_LLM_CHECK_EQ(suite, snapshot.metrics.released_total, 2U);
    });

    suite.run("HTTP callers submit concurrently without owning ServiceCore", [&] {
        GenerationConfig config;
        config.max_new_tokens = 2;
        Transformer transformer(2, Embedding(32), PositionEncoding(16));
        Runtime runtime(config, transformer);
        GreedySampler sampler;
        ServiceCore service(8, 4);
        AsyncServiceLoop loop(service, runtime, sampler, 8);
        HttpAdapter adapter(service, runtime, sampler, {}, {}, {}, &loop);
        HttpResponse first;
        HttpResponse second;
        std::thread first_client([&] {
            first = adapter.handle({
                "POST", "/v1/completions",
                "{\"request_id\":30,\"prompt_tokens\":[1,2]}"});
        });
        std::thread second_client([&] {
            second = adapter.handle({
                "POST", "/v1/completions",
                "{\"request_id\":31,\"prompt_tokens\":[3,4]}"});
        });
        first_client.join();
        second_client.join();
        CPP_LLM_CHECK_EQ(suite, first.status_code, 200);
        CPP_LLM_CHECK_EQ(suite, second.status_code, 200);
        CPP_LLM_CHECK(
            suite, first.body.find("finish_reason\":\"stop") != std::string::npos);
        CPP_LLM_CHECK(
            suite, second.body.find("finish_reason\":\"stop") != std::string::npos);

        RecordingWriter writer;
        const HttpResponse streamed = adapter.handle_stream({
            "POST", "/v1/completions",
            "{\"request_id\":32,\"prompt_tokens\":[5]}"}, writer);
        CPP_LLM_CHECK_EQ(suite, streamed.status_code, 200);
        CPP_LLM_CHECK(suite, writer.headers && writer.finished);
        CPP_LLM_CHECK(
            suite, writer.body.find("\"step\":0") != std::string::npos);
        CPP_LLM_CHECK(
            suite, writer.body.find("finish_reason\":\"stop") != std::string::npos);
        const HttpResponse metrics = adapter.handle({"GET", "/metrics", {}});
        CPP_LLM_CHECK(
            suite, metrics.body.find("serving_completed_total 3") != std::string::npos);
    });

    suite.run("collection window admits one submission batch", [&] {
        GenerationConfig config;
        config.max_new_tokens = 1;
        Transformer transformer(2, Embedding(32), PositionEncoding(16));
        Runtime runtime(config, transformer);
        GreedySampler sampler;
        ServiceCore service(4, 4);
        AsyncServiceLoop loop(
            service, runtime, sampler, 8, 0, std::chrono::milliseconds(5));

        const AsyncRequestHandle first = loop.submit(40, {1, 2});
        const AsyncRequestHandle second = loop.submit(41, {3, 4});
        CPP_LLM_CHECK(
            suite, first.wait().status == AsyncCompletionStatus::completed);
        CPP_LLM_CHECK(
            suite, second.wait().status == AsyncCompletionStatus::completed);
        const AsyncServiceSnapshot snapshot = loop.snapshot();
        CPP_LLM_CHECK_EQ(suite, snapshot.submission_batches, 1U);
        CPP_LLM_CHECK_EQ(suite, snapshot.max_submission_batch_size, 2U);
    });

    return suite.finish();
}
