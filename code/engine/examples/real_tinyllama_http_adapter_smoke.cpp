// Engine/Chapter 10：真实 TinyLlama HTTP adapter 组合 smoke。
// 职责：验证同步 completion 与 SSE 共用 ServiceCore，并按请求创建独立 Executor。
// 运行：real_tinyllama_http_adapter_smoke <model.gguf>。
#include "cpp_llm_engine/real_llama_executor_factory.h"
#include "cpp_llm_engine/real_tokenizer_backend.h"
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

std::string completion_body(
    std::size_t request_id,
    const chapter01::TokenSequence& prompt_tokens) {
    std::ostringstream body;
    body << "{\"request_id\":" << request_id << ",\"prompt_tokens\":[";
    for (std::size_t index = 0; index < prompt_tokens.size(); ++index) {
        if (index > 0) body << ',';
        body << prompt_tokens[index];
    }
    body << "]}";
    return body.str();
}

bool contains_in_order(
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
        std::cerr << "usage: real_tinyllama_http_adapter_smoke <model.gguf>\n";
        return 2;
    }

    std::string error;
    auto tokenizer = cpp_llm_engine::RealTokenizerBackend::load(argv[1], &error);
    auto factory = cpp_llm_engine::RealLlamaExecutorFactory::load(argv[1], &error);
    if (tokenizer == nullptr || factory == nullptr) {
        std::cerr << "real HTTP composition load failed: " << error << '\n';
        return 1;
    }
    if (tokenizer->vocab_size() != factory->vocab_size()) {
        std::cerr << "tokenizer and model vocabulary sizes differ\n";
        return 1;
    }

    chapter06_generation::GenerationConfig config;
    config.max_new_tokens = 4;
    config.eos_token_id = factory->eos_token_id();
    chapter06_generation::Runtime runtime(config);
    chapter06_generation::GreedySampler sampler;
    if (!factory->configure_paged_kv_cache(8, 4, &error)) {
        std::cerr << "paged KV setup failed: " << error << '\n';
        return 1;
    }
    chapter10_serving::ServiceCore service(
        2, 2, {}, factory->paged_kv_resource(), {}, factory.get());
    chapter10_serving::HttpAdapter adapter(
        service,
        runtime,
        sampler,
        [&factory]() { return factory->create_executor(); });

    const auto sync_prompt = tokenizer->encode("Hello TinyLlama");
    const chapter10_serving::HttpResponse sync_response = adapter.handle({
        "POST", "/v1/completions", completion_body(101, sync_prompt)});
    const std::string expected_sync =
        "\"generated_token_ids\":[29892,306,29915,29885]";
    const bool sync_golden_match =
        sync_response.status_code == 200 &&
        sync_response.body.find(expected_sync) != std::string::npos;

    const auto stream_prompt = tokenizer->encode("Hi");
    RecordingWriter writer;
    const chapter10_serving::HttpResponse stream_response = adapter.handle_stream({
        "POST", "/v1/completions", completion_body(102, stream_prompt)}, writer);
    const chapter06_generation::TokenSequence expected_stream{839, 744, 29892, 263};
    const bool stream_golden_match =
        stream_response.status_code == 200 && writer.headers_written && writer.finished &&
        writer.head.status_code == 200 &&
        writer.head.content_type == "text/event-stream" &&
        contains_in_order(writer.body, expected_stream) &&
        writer.body.find("\"finish_reason\":\"stop\"") != std::string::npos;

    const chapter10_serving::HttpResponse metrics =
        adapter.handle({"GET", "/metrics", {}});
    const bool metrics_match =
        metrics.status_code == 200 &&
        metrics.body.find("serving_accepted_total 2\n") != std::string::npos &&
        metrics.body.find("serving_completed_total 2\n") != std::string::npos &&
        metrics.body.find("serving_failed_total 0\n") != std::string::npos &&
        metrics.body.find("serving_released_total 2\n") != std::string::npos &&
        metrics.body.find("serving_generated_tokens_total 8\n") != std::string::npos &&
        metrics.body.find("serving_queue_wait_seconds_count 2\n") != std::string::npos &&
        metrics.body.find("serving_ttft_seconds_count 2\n") != std::string::npos &&
        metrics.body.find("serving_completion_seconds_count 2\n") != std::string::npos;

    std::cout << "sync_golden_match: " << std::boolalpha << sync_golden_match << '\n'
              << "stream_golden_match: " << stream_golden_match << '\n'
              << "metrics_match: " << metrics_match << '\n'
              << "real_http_adapter: ready\n";
    return sync_golden_match && stream_golden_match && metrics_match ? 0 : 1;
}
