// Chapter 09：实现 transport-neutral HTTP 到 ServiceCore 的同步映射。
// 契约：只解析教程使用的最小 JSON body，不创建 socket，也不复制调度状态。
// 验证：HTTP adapter 测试检查 endpoint、状态码、完成结果和 metrics 字段。
#include "chapter09_serving/http_adapter.h"

#include "chapter06_generation/request.h"

#include <cctype>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>

namespace chapter09_serving {

namespace {

HttpResponse json_response(int status, const std::string& body) {
    return {status, "application/json", body};
}

bool read_number(const std::string& body, const std::string& key, std::size_t& value) {
    const std::string marker = "\"" + key + "\"";
    const std::size_t key_position = body.find(marker);
    if (key_position == std::string::npos) return false;
    const std::size_t colon = body.find(':', key_position + marker.size());
    if (colon == std::string::npos) return false;
    std::size_t position = colon + 1;
    while (position < body.size() && std::isspace(static_cast<unsigned char>(body[position]))) ++position;
    const std::size_t start = position;
    while (position < body.size() && std::isdigit(static_cast<unsigned char>(body[position]))) ++position;
    if (start == position) return false;
    value = static_cast<std::size_t>(std::stoull(body.substr(start, position - start)));
    return true;
}

bool read_tokens(const std::string& body, chapter06_generation::TokenSequence& tokens) {
    const std::size_t key_position = body.find("\"prompt_tokens\"");
    if (key_position == std::string::npos) return false;
    const std::size_t open = body.find('[', key_position);
    const std::size_t close = body.find(']', open);
    if (open == std::string::npos || close == std::string::npos) return false;
    std::size_t position = open + 1;
    while (position < close) {
        while (position < close && (std::isspace(static_cast<unsigned char>(body[position])) || body[position] == ',')) ++position;
        if (position == close) break;
        const std::size_t start = position;
        while (position < close && std::isdigit(static_cast<unsigned char>(body[position]))) ++position;
        if (start == position) return false;
        tokens.push_back(static_cast<chapter06_generation::TokenId>(
            std::stoull(body.substr(start, position - start))));
    }
    return !tokens.empty();
}

bool read_bool(const std::string& body, const std::string& key, bool& value) {
    const std::string marker = "\"" + key + "\"";
    const std::size_t key_position = body.find(marker);
    if (key_position == std::string::npos) return false;
    const std::size_t colon = body.find(':', key_position + marker.size());
    if (colon == std::string::npos) return false;
    std::size_t position = colon + 1;
    while (position < body.size() && std::isspace(static_cast<unsigned char>(body[position]))) ++position;
    if (body.compare(position, 4, "true") == 0) {
        value = true;
        return true;
    }
    if (body.compare(position, 5, "false") == 0) {
        value = false;
        return true;
    }
    return false;
}

}  // namespace

HttpAdapter::HttpAdapter(
    ServingCore& service,
    chapter06_generation::Runtime& runtime,
    const chapter06_generation::Sampler& sampler,
    RequestExecutorFactory executor_factory)
    : service_(service), runtime_(runtime), sampler_(sampler),
      executor_factory_(std::move(executor_factory)) {}

HttpResponse HttpAdapter::handle(const HttpRequest& request) {
    if (request.method == "GET" && request.path == "/health") {
        return json_response(200, "{\"status\":\"ok\"}");
    }
    if (request.method == "GET" && request.path == "/metrics") {
        return handle_metrics();
    }
    if (request.method == "POST" && request.path == "/v1/completions") {
        return handle_completion(request);
    }
    return json_response(404, "{\"error\":\"route_not_found\"}");
}

HttpResponse HttpAdapter::handle_stream(
    const HttpRequest& request,
    ResponseWriter& writer) {
    if (request.method != "POST" || request.path != "/v1/completions") {
        return json_response(404, "{\"error\":\"route_not_found\"}");
    }
    return handle_stream_completion(request, writer);
}

HttpResponse HttpAdapter::handle_stream_completion(
    const HttpRequest& request,
    ResponseWriter& writer) {
    std::size_t request_id = 0;
    chapter06_generation::TokenSequence prompt;
    if (!read_number(request.body, "request_id", request_id) ||
        !read_tokens(request.body, prompt)) {
        return json_response(400, "{\"error\":\"invalid_completion_body\"}");
    }
    if (requests_.find(request_id) != requests_.end()) {
        return json_response(409, "{\"error\":\"duplicate_request_id\"}");
    }

    auto owned_request = std::make_unique<chapter06_generation::Request>(std::move(prompt));
    std::shared_ptr<chapter06_generation::RequestModelExecutor> executor;
    if (executor_factory_) {
        executor = executor_factory_();
        if (executor == nullptr) {
            return json_response(500, "{\"error\":\"model_executor_unavailable\"}");
        }
    }
    const AdmissionResult admission = service_.admit(
        request_id, *owned_request, sampler_, std::move(executor));
    if (!admission.accepted()) {
        if (admission.status == AdmissionStatus::QueueFull) {
            return json_response(429, "{\"error\":\"queue_full\"}");
        }
        return json_response(400, "{\"error\":\"request_rejected\"}");
    }
    chapter06_generation::Request* request_ptr = owned_request.get();
    requests_.emplace(request_id, std::move(owned_request));

    if (!request.client_connected) {
        service_.cancel(request_id);
        requests_.erase(request_id);
        return {499, "text/event-stream", {}};
    }
    if (!writer.write_headers({200, "text/event-stream"})) {
        const bool connected = writer.client_connected();
        service_.cancel(request_id);
        requests_.erase(request_id);
        return {connected ? 503 : 499, "text/event-stream", {}};
    }

    bool write_failed = false;
    WriteResult write_result = WriteResult::Ok;
    const ServiceStats stats = service_.dispatch(
        runtime_,
        [&](std::size_t callback_request_id, chapter06_generation::TokenId token, std::size_t step) {
            if (callback_request_id != request_id || write_failed) return;
            std::ostringstream event;
            event << "data: {\"token\":" << token << ",\"step\":" << step << "}\n\n";
            write_result = writer.write_event(event.str());
            if (write_result != WriteResult::Ok) {
                write_failed = true;
                service_.cancel(request_id);
                return;
            }
            const std::size_t generated = request_ptr->generated_size();
            if (request.disconnect_after_tokens > 0 &&
                generated >= request.disconnect_after_tokens) {
                service_.cancel(request_id);
            } else if (request.timeout_after_tokens > 0 &&
                       generated >= request.timeout_after_tokens) {
                service_.timeout(request_id);
            }
        });
    (void)stats;
    service_.release_completed();

    const bool cancelled = request_ptr->cancelled();
    const bool timed_out = request_ptr->timed_out();
    int semantic_status = 200;
    if (!write_failed && writer.client_connected()) {
        if (cancelled) {
            writer.write_event("data: {\"finish_reason\":\"client_disconnect\"}\n\n");
            semantic_status = 499;
        } else if (timed_out) {
            writer.write_event("data: {\"finish_reason\":\"timeout\"}\n\n");
            semantic_status = 408;
        } else if (request_ptr->finished()) {
            writer.write_event("data: {\"finish_reason\":\"stop\"}\n\n");
        } else {
            writer.write_event("data: {\"finish_reason\":\"runtime_error\"}\n\n");
            semantic_status = 500;
        }
    } else if (write_result == WriteResult::Backpressure) {
        semantic_status = 503;
    } else {
        semantic_status = 499;
    }
    writer.finish();
    requests_.erase(request_id);
    return {semantic_status, "text/event-stream", {}};
}

HttpResponse HttpAdapter::handle_completion(const HttpRequest& request) {
    std::size_t request_id = 0;
    chapter06_generation::TokenSequence prompt;
    if (!read_number(request.body, "request_id", request_id) ||
        !read_tokens(request.body, prompt)) {
        return json_response(400, "{\"error\":\"invalid_completion_body\"}");
    }
    bool stream = false;
    (void)read_bool(request.body, "stream", stream);
    if (requests_.find(request_id) != requests_.end()) {
        return json_response(409, "{\"error\":\"duplicate_request_id\"}");
    }

    auto owned_request = std::make_unique<chapter06_generation::Request>(std::move(prompt));
    std::shared_ptr<chapter06_generation::RequestModelExecutor> executor;
    if (executor_factory_) {
        executor = executor_factory_();
        if (executor == nullptr) {
            return json_response(500, "{\"error\":\"model_executor_unavailable\"}");
        }
    }
    const AdmissionResult admission = service_.admit(
        request_id, *owned_request, sampler_, std::move(executor));
    if (!admission.accepted()) {
        if (admission.status == AdmissionStatus::QueueFull) {
            return json_response(429, "{\"error\":\"queue_full\"}");
        }
        return json_response(400, "{\"error\":\"request_rejected\"}");
    }
    chapter06_generation::Request* request_ptr = owned_request.get();
    requests_.emplace(request_id, std::move(owned_request));
    if (!request.client_connected) {
        service_.cancel(request_id);
    }

    std::vector<chapter06_generation::TokenId> streamed_tokens;
    const ServiceStats stats = service_.dispatch(
        runtime_,
        [&](std::size_t /*callback_request_id*/, chapter06_generation::TokenId token, std::size_t /*step*/) {
            streamed_tokens.push_back(token);
            if (request.disconnect_after_tokens > 0 &&
                streamed_tokens.size() >= request.disconnect_after_tokens) {
                service_.cancel(request_id);
            } else if (request.timeout_after_tokens > 0 &&
                       streamed_tokens.size() >= request.timeout_after_tokens) {
                service_.timeout(request_id);
            }
        });
    service_.release_completed();
    const bool finished = request_ptr->finished();
    const std::size_t generated = request_ptr->generated_size();
    const chapter06_generation::TokenSequence generated_tokens =
        request_ptr->generated();
    const bool cancelled = request_ptr->cancelled();
    const bool timed_out = request_ptr->timed_out();
    if (stream) {
        std::ostringstream body;
        for (std::size_t index = 0; index < streamed_tokens.size(); ++index) {
            body << "data: {\"token\":" << streamed_tokens[index]
                 << ",\"step\":" << index << "}\n\n";
        }
        if (cancelled) {
            body << "data: {\"finish_reason\":\"client_disconnect\"}\n\n";
            requests_.erase(request_id);
            return {499, "text/event-stream", body.str()};
        }
        if (timed_out) {
            body << "data: {\"finish_reason\":\"timeout\"}\n\n";
            requests_.erase(request_id);
            return {408, "text/event-stream", body.str()};
        }
        if (!finished) {
            requests_.erase(request_id);
            return {500, "text/event-stream", "data: {\"finish_reason\":\"runtime_error\"}\n\n"};
        }
        body << "data: {\"finish_reason\":\"stop\"}\n\n";
        requests_.erase(request_id);
        return {200, "text/event-stream", body.str()};
    }
    if (cancelled) {
        requests_.erase(request_id);
        return json_response(499, "{\"error\":\"client_disconnect\"}");
    }
    if (timed_out) {
        requests_.erase(request_id);
        return json_response(408, "{\"error\":\"timeout\"}");
    }
    requests_.erase(request_id);
    if (!finished) {
        return json_response(500, "{\"error\":\"runtime_did_not_finish\"}");
    }

    std::ostringstream body;
    body << "{\"request_id\":" << request_id
         << ",\"generated_tokens\":" << generated
         << ",\"generated_token_ids\":[";
    for (std::size_t index = 0; index < generated_tokens.size(); ++index) {
        if (index > 0) body << ',';
        body << generated_tokens[index];
    }
    body << ']'
         << ",\"batch_generated_tokens\":" << stats.generated_tokens
         << ",\"finish_reason\":\"stop\"}";
    return json_response(200, body.str());
}

HttpResponse HttpAdapter::handle_metrics() const {
    const ServiceMetrics& metrics = service_.metrics();
    std::ostringstream body;
    body << "# TYPE serving_accepted_total counter\n"
         << "serving_accepted_total " << metrics.accepted_total << '\n'
         << "serving_queue_full_total " << metrics.queue_full_total << '\n'
         << "serving_duplicate_total " << metrics.duplicate_total << '\n'
         << "serving_invalid_total " << metrics.invalid_total << '\n'
         << "serving_cancelled_total " << metrics.cancelled_total << '\n'
         << "serving_timed_out_total " << metrics.timed_out_total << '\n'
         << "serving_failed_total " << metrics.failed_total << '\n'
         << "serving_completed_total " << metrics.completed_total << '\n'
         << "serving_released_total " << metrics.released_total << '\n'
         << "serving_generated_tokens_total " << metrics.generated_tokens_total << '\n'
         << "serving_peak_inflight " << metrics.peak_inflight << '\n'
         << "# TYPE serving_queue_wait_seconds summary\n"
         << "serving_queue_wait_seconds_sum " << metrics.queue_wait_seconds_sum << '\n'
         << "serving_queue_wait_seconds_count " << metrics.queue_wait_count << '\n'
         << "# TYPE serving_ttft_seconds summary\n"
         << "serving_ttft_seconds_sum " << metrics.ttft_seconds_sum << '\n'
         << "serving_ttft_seconds_count " << metrics.ttft_count << '\n'
         << "# TYPE serving_completion_seconds summary\n"
         << "serving_completion_seconds_sum " << metrics.completion_seconds_sum << '\n'
         << "serving_completion_seconds_count " << metrics.completion_count << '\n';
    return {200, "text/plain; version=0.0.4", body.str()};
}

}  // namespace chapter09_serving
