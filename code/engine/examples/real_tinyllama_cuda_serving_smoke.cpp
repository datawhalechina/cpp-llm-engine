// Engine/Chapter 10：真实 TinyLlama CUDA 服务组合验收。
// 职责：验证 HTTP/SSE 契约可直接注入 CUDA Executor，并统一释放四类请求终态。
// 运行：real_tinyllama_cuda_serving_smoke <model.gguf>。
#include "cpp_llm_engine/real_llama_cuda_executor_factory.h"
#include "cpp_llm_engine/real_tokenizer_backend.h"
#include "chapter06_generation/request.h"
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

bool metric_is(const std::string& metrics, const std::string& name, std::size_t value) {
    return metrics.find(name + " " + std::to_string(value) + "\n") != std::string::npos;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "usage: real_tinyllama_cuda_serving_smoke <model.gguf>\n";
        return 2;
    }

    std::string error;
    auto tokenizer = cpp_llm_engine::RealTokenizerBackend::load(argv[1], &error);
    auto factory = cpp_llm_engine::RealLlamaCudaExecutorFactory::load(argv[1], 64, &error);
    if (tokenizer == nullptr || factory == nullptr) {
        std::cerr << "CUDA service load failed: " << error << '\n';
        return 1;
    }
    if (tokenizer->vocab_size() != factory->vocab_size()) {
        std::cerr << "tokenizer and CUDA model vocabulary sizes differ\n";
        return 1;
    }
    if (!factory->configure_paged_kv_cache(16, 2, &error)) {
        std::cerr << "CUDA paged KV setup failed: " << error << '\n';
        return 1;
    }

    chapter06_generation::GenerationConfig config;
    config.max_new_tokens = 4;
    config.eos_token_id = factory->eos_token_id();
    chapter06_generation::Runtime runtime(config);
    chapter06_generation::GreedySampler sampler;
    chapter10_serving::ServiceCore service(2, 2);
    chapter10_serving::HttpAdapter adapter(
        service,
        runtime,
        sampler,
        [&factory]() { return factory->create_executor(); });

    const auto hello_prompt = tokenizer->encode("Hello TinyLlama");
    const auto hi_prompt = tokenizer->encode("Hi");

    chapter06_generation::Request batch_request_a(hello_prompt);
    chapter06_generation::Request batch_request_b(hi_prompt);
    chapter10_serving::ServiceCore batch_service(
        2, 2, {}, factory->paged_kv_resource(), {}, factory.get());
    const bool batch_admitted =
        batch_service.admit(
            201, batch_request_a, sampler, factory->create_executor()).accepted() &&
        batch_service.admit(
            202, batch_request_b, sampler, factory->create_executor()).accepted();
    std::size_t model_batch_calls = 0;
    std::size_t tensor_batch_requests = 0;
    for (std::size_t step = 0;
         step < 16 && (!batch_request_a.terminal() || !batch_request_b.terminal());
         ++step) {
        const auto stats = batch_service.dispatch_step(runtime, 8);
        model_batch_calls += stats.model_batch_calls;
        tensor_batch_requests += stats.tensor_batch_requests;
        if (stats.scheduled_requests == 0) break;
    }
    const bool scheduler_batch_match =
        batch_admitted && batch_request_a.finished() && batch_request_b.finished() &&
        batch_request_a.generated() ==
            chapter06_generation::TokenSequence({29892, 306, 29915, 29885}) &&
        batch_request_b.generated() ==
            chapter06_generation::TokenSequence({839, 744, 29892, 263}) &&
        model_batch_calls > 0 && tensor_batch_requests >= 2;
    const bool batch_resources_released =
        batch_service.release_terminal() == 2 && batch_service.inflight_size() == 0;

    const auto sync_response = adapter.handle({
        "POST", "/v1/completions", completion_body(101, hello_prompt)});
    const bool sync_golden_match =
        sync_response.status_code == 200 &&
        sync_response.body.find(
            "\"generated_token_ids\":[29892,306,29915,29885]") != std::string::npos;

    RecordingWriter writer;
    const auto stream_response = adapter.handle_stream({
        "POST", "/v1/completions", completion_body(102, hi_prompt)}, writer);
    const chapter06_generation::TokenSequence expected_stream{839, 744, 29892, 263};
    const bool stream_golden_match =
        stream_response.status_code == 200 && writer.headers_written && writer.finished &&
        writer.head.status_code == 200 &&
        writer.head.content_type == "text/event-stream" &&
        contains_in_order(writer.body, expected_stream) &&
        writer.body.find("\"finish_reason\":\"stop\"") != std::string::npos;

    chapter10_serving::HttpRequest cancelled{
        "POST", "/v1/completions", completion_body(103, hi_prompt)};
    cancelled.disconnect_after_tokens = 1;
    const auto cancelled_response = adapter.handle(cancelled);

    chapter10_serving::HttpRequest timed_out{
        "POST", "/v1/completions", completion_body(104, hi_prompt)};
    timed_out.timeout_after_tokens = 1;
    const auto timeout_response = adapter.handle(timed_out);
    const bool terminal_status_match =
        cancelled_response.status_code == 499 && timeout_response.status_code == 408;

    const auto metrics = adapter.handle({"GET", "/metrics", {}});
    const bool metrics_match =
        metrics.status_code == 200 &&
        metric_is(metrics.body, "serving_accepted_total", 4) &&
        metric_is(metrics.body, "serving_completed_total", 2) &&
        metric_is(metrics.body, "serving_cancelled_total", 1) &&
        metric_is(metrics.body, "serving_timed_out_total", 1) &&
        metric_is(metrics.body, "serving_failed_total", 0) &&
        metric_is(metrics.body, "serving_released_total", 4) &&
        metric_is(metrics.body, "serving_generated_tokens_total", 10) &&
        metric_is(metrics.body, "serving_inflight", 0) &&
        metric_is(metrics.body, "serving_waiting", 0) &&
        metric_is(metrics.body, "serving_running", 0);

    std::cout << "sync_golden_match: " << std::boolalpha << sync_golden_match << '\n'
              << "stream_golden_match: " << stream_golden_match << '\n'
              << "terminal_status_match: " << terminal_status_match << '\n'
              << "metrics_match: " << metrics_match << '\n'
              << "scheduler_batch_match: " << scheduler_batch_match << '\n'
              << "batch_resources_released: " << batch_resources_released << '\n'
              << "cuda_serving: ready\n";
    return sync_golden_match && stream_golden_match && terminal_status_match &&
        metrics_match && scheduler_batch_match && batch_resources_released ? 0 : 1;
}
