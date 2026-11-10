// Engine/Chapter10：验证 DeepSeek-V2-Lite 接入通用 HTTP/SSE 适配层。
#include "cpp_llm_engine/real_deepseek2_executor_factory.h"

#include "chapter06_generation/runtime.h"
#include "chapter06_generation/sampler.h"
#include "chapter10_serving/http_adapter.h"

#include <iostream>
#include <sstream>
#include <string>
#include <string_view>

namespace {

class RecordingWriter final : public chapter10_serving::ResponseWriter {
public:
    bool write_headers(const chapter10_serving::ResponseHead& value) override {
        head = value;
        headers_written = true;
        return true;
    }

    chapter10_serving::WriteResult write_event(std::string_view event) override {
        body += event;
        return chapter10_serving::WriteResult::ok;
    }

    void finish() override { finished = true; }
    bool client_connected() const override { return true; }

    chapter10_serving::ResponseHead head;
    std::string body;
    bool headers_written = false;
    bool finished = false;
};

std::string completion_body(std::size_t request_id) {
    return "{\"request_id\":" + std::to_string(request_id) +
        ",\"prompt_tokens\":[0]}";
}

bool contains_tokens_in_order(
    const std::string& text,
    const chapter06_generation::TokenSequence& tokens) {
    std::size_t position = 0;
    for (const auto token : tokens) {
        const std::string marker = "\"token\":" + std::to_string(token);
        position = text.find(marker, position);
        if (position == std::string::npos) return false;
        position += marker.size();
    }
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "usage: real_deepseek2_http_adapter_smoke <model.gguf>\n";
        return 2;
    }
    std::string error;
    auto factory = cpp_llm_engine::RealDeepSeek2ExecutorFactory::load(
        argv[1], &error);
    if (factory == nullptr) {
        std::cerr << "DeepSeek-V2-Lite load failed: " << error << '\n';
        return 1;
    }

    chapter06_generation::GenerationConfig config;
    config.max_new_tokens = 2;
    config.eos_token_id = factory->eos_token_id();
    chapter06_generation::Runtime runtime(config);
    chapter06_generation::GreedySampler sampler;
    chapter10_serving::ServiceCore service(2, 2, {}, {}, {}, factory.get());
    chapter10_serving::HttpAdapter adapter(
        service,
        runtime,
        sampler,
        [&factory]() { return factory->create_executor(); });

    const chapter06_generation::TokenSequence golden{185, 85770};
    const auto completion = adapter.handle({
        "POST", "/v1/completions", completion_body(201)});
    const bool completion_match = completion.status_code == 200 &&
        completion.body.find("\"generated_token_ids\":[185,85770]") !=
            std::string::npos &&
        completion.body.find("\"finish_reason\":\"stop\"") !=
            std::string::npos;

    RecordingWriter writer;
    const auto stream = adapter.handle_stream({
        "POST", "/v1/completions", completion_body(202)}, writer);
    const bool stream_match = stream.status_code == 200 &&
        writer.headers_written && writer.finished &&
        writer.head.status_code == 200 &&
        writer.head.content_type == "text/event-stream" &&
        contains_tokens_in_order(writer.body, golden) &&
        writer.body.find("\"finish_reason\":\"stop\"") != std::string::npos;

    const auto metrics = adapter.handle({"GET", "/metrics", {}});
    const bool lifecycle_match = metrics.status_code == 200 &&
        metrics.body.find("serving_accepted_total 2\n") != std::string::npos &&
        metrics.body.find("serving_completed_total 2\n") != std::string::npos &&
        metrics.body.find("serving_failed_total 0\n") != std::string::npos &&
        metrics.body.find("serving_released_total 2\n") != std::string::npos &&
        metrics.body.find("serving_generated_tokens_total 4\n") !=
            std::string::npos &&
        service.inflight_size() == 0;

    const bool passed = completion_match && stream_match && lifecycle_match;
    std::cout << "completion_match: " << std::boolalpha << completion_match
              << "\nstream_match: " << stream_match
              << "\nlifecycle_match: " << lifecycle_match
              << "\ndeepseek2_http_adapter: " << (passed ? "PASS" : "FAIL")
              << '\n';
    return passed ? 0 : 1;
}
