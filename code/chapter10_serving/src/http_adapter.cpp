// Chapter 10：实现最小 HTTP/SSE 到 ServiceCore 的同步映射。
// 契约：Adapter 在销毁 Request 前必须先让 ServiceCore 完成统一终态释放。
// 验证：chapter10_http_contract_test 检查正常、断连、背压与 metrics。
#include "chapter10_serving/http_adapter.h"

#include "chapter06_generation/request.h"
#include "chapter06_generation/runtime.h"
#include "chapter06_generation/sampler.h"

#include <cctype>
#include <charconv>
#include <sstream>
#include <utility>
#include <vector>

namespace chapter10_serving {

namespace {

HttpResponse json_response(int status, const std::string& body) {
    return {status, "application/json", body};
}

bool read_number(const std::string& body, const std::string& key, std::size_t* value) {
    if (value == nullptr) return false;
    const std::string marker = "\"" + key + "\"";
    const std::size_t key_position = body.find(marker);
    if (key_position == std::string::npos) return false;
    const std::size_t colon = body.find(':', key_position + marker.size());
    if (colon == std::string::npos) return false;
    std::size_t position = colon + 1;
    while (position < body.size() &&
           std::isspace(static_cast<unsigned char>(body[position]))) {
        ++position;
    }
    const std::size_t start = position;
    while (position < body.size() &&
           std::isdigit(static_cast<unsigned char>(body[position]))) {
        ++position;
    }
    if (start == position) return false;
    const char* begin = body.data() + start;
    const char* end = body.data() + position;
    const auto parsed = std::from_chars(begin, end, *value);
    return parsed.ec == std::errc{} && parsed.ptr == end;
}

bool read_tokens(
    const std::string& body,
    chapter06_generation::TokenSequence* tokens) {
    if (tokens == nullptr) return false;
    tokens->clear();
    const std::size_t key_position = body.find("\"prompt_tokens\"");
    if (key_position == std::string::npos) return false;
    const std::size_t open = body.find('[', key_position);
    const std::size_t close = body.find(']', open);
    if (open == std::string::npos || close == std::string::npos) return false;
    std::size_t position = open + 1;
    while (position < close) {
        while (position < close &&
               (std::isspace(static_cast<unsigned char>(body[position])) ||
                body[position] == ',')) {
            ++position;
        }
        if (position == close) break;
        const std::size_t start = position;
        while (position < close &&
               std::isdigit(static_cast<unsigned char>(body[position]))) {
            ++position;
        }
        if (start == position) return false;
        chapter06_generation::TokenId token = 0;
        const char* begin = body.data() + start;
        const char* end = body.data() + position;
        const auto parsed = std::from_chars(begin, end, token);
        if (parsed.ec != std::errc{} || parsed.ptr != end) return false;
        tokens->push_back(token);
    }
    return !tokens->empty();
}

HttpResponse admission_error(AdmissionStatus status) {
    switch (status) {
        case AdmissionStatus::queue_full:
            return json_response(429, "{\"error\":\"queue_full\"}");
        case AdmissionStatus::duplicate_id:
            return json_response(409, "{\"error\":\"duplicate_request_id\"}");
        case AdmissionStatus::invalid_request:
            return json_response(400, "{\"error\":\"invalid_request\"}");
        case AdmissionStatus::accepted:
            break;
    }
    return json_response(500, "{\"error\":\"unexpected_admission_status\"}");
}

}  // namespace

HttpAdapter::HttpAdapter(
    ServiceCore& service,
    chapter06_generation::Runtime& runtime,
    const chapter06_generation::Sampler& sampler,
    RequestExecutorFactory executor_factory)
    : service_(service),
      runtime_(runtime),
      sampler_(sampler),
      executor_factory_(std::move(executor_factory)) {}

HttpAdapter::~HttpAdapter() = default;

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

HttpResponse HttpAdapter::handle_completion(const HttpRequest& request) {
    std::size_t request_id = 0;
    chapter06_generation::TokenSequence prompt;
    if (!read_number(request.body, "request_id", &request_id) ||
        !read_tokens(request.body, &prompt)) {
        return json_response(400, "{\"error\":\"invalid_completion_body\"}");
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
    if (!admission.accepted()) return admission_error(admission.status);

    chapter06_generation::Request* request_ptr = owned_request.get();
    requests_.emplace(request_id, std::move(owned_request));
    if (!request.client_connected) service_.cancel(request_id);

    try {
        while (!request_ptr->terminal()) {
            const DispatchStats stats = service_.dispatch_step(
                runtime_, 1024,
                [&](std::size_t callback_id, chapter06_generation::TokenId, std::size_t) {
                    if (callback_id != request_id) return;
                    if (request.disconnect_after_tokens > 0 &&
                        request_ptr->generated_size() >= request.disconnect_after_tokens) {
                        service_.cancel(request_id);
                    } else if (request.timeout_after_tokens > 0 &&
                               request_ptr->generated_size() >= request.timeout_after_tokens) {
                        service_.timeout(request_id);
                    }
                });
            if (stats.scheduled_requests == 0 && !request_ptr->terminal()) {
                request_ptr->mark_failed("service dispatch made no progress");
            }
        }
    } catch (...) {
        if (!request_ptr->terminal()) {
            request_ptr->mark_failed("service dispatch threw an exception");
        }
    }

    const auto generated = request_ptr->generated();
    const bool finished = request_ptr->finished();
    const bool cancelled = request_ptr->cancelled();
    const bool timed_out = request_ptr->timed_out();
    service_.release_terminal();
    requests_.erase(request_id);
    if (cancelled) return json_response(499, "{\"error\":\"client_disconnect\"}");
    if (timed_out) return json_response(408, "{\"error\":\"timeout\"}");
    if (!finished) {
        return json_response(500, "{\"error\":\"runtime_did_not_finish\"}");
    }

    std::ostringstream body;
    body << "{\"request_id\":" << request_id << ",\"generated_token_ids\":[";
    for (std::size_t index = 0; index < generated.size(); ++index) {
        if (index > 0) body << ',';
        body << generated[index];
    }
    body << "],\"finish_reason\":\"stop\"}";
    return json_response(200, body.str());
}

HttpResponse HttpAdapter::handle_stream(
    const HttpRequest& request,
    ResponseWriter& writer) {
    if (request.method != "POST" || request.path != "/v1/completions") {
        return json_response(404, "{\"error\":\"route_not_found\"}");
    }
    std::size_t request_id = 0;
    chapter06_generation::TokenSequence prompt;
    if (!read_number(request.body, "request_id", &request_id) ||
        !read_tokens(request.body, &prompt)) {
        return json_response(400, "{\"error\":\"invalid_completion_body\"}");
    }

    auto owned_request = std::make_unique<chapter06_generation::Request>(std::move(prompt));
    std::shared_ptr<chapter06_generation::RequestModelExecutor> executor;
    if (executor_factory_) executor = executor_factory_();
    if (executor_factory_ && executor == nullptr) {
        return json_response(500, "{\"error\":\"model_executor_unavailable\"}");
    }
    const AdmissionResult admission = service_.admit(
        request_id, *owned_request, sampler_, std::move(executor));
    if (!admission.accepted()) return admission_error(admission.status);

    chapter06_generation::Request* request_ptr = owned_request.get();
    requests_.emplace(request_id, std::move(owned_request));
    WriteResult write_result = WriteResult::ok;
    bool adapter_exception = false;
    try {
        if (!request.client_connected) {
            write_result = WriteResult::client_closed;
            service_.cancel(request_id);
        } else if (!writer.write_headers({200, "text/event-stream"})) {
            write_result = writer.client_connected()
                               ? WriteResult::backpressure
                               : WriteResult::client_closed;
            service_.cancel(request_id);
        }

        while (!request_ptr->terminal()) {
            const DispatchStats stats = service_.dispatch_step(
                runtime_, 1024,
                [&](std::size_t callback_id,
                    chapter06_generation::TokenId token,
                    std::size_t step) {
                    if (callback_id != request_id || write_result != WriteResult::ok) return;
                    std::ostringstream event;
                    event << "data: {\"token\":" << token << ",\"step\":" << step
                          << "}\n\n";
                    write_result = writer.write_event(event.str());
                    if (write_result != WriteResult::ok) {
                        service_.cancel(request_id);
                    } else if (request.disconnect_after_tokens > 0 &&
                               request_ptr->generated_size() >=
                                   request.disconnect_after_tokens) {
                        service_.cancel(request_id);
                    } else if (request.timeout_after_tokens > 0 &&
                               request_ptr->generated_size() >=
                                   request.timeout_after_tokens) {
                        service_.timeout(request_id);
                    }
                });
            if (stats.scheduled_requests == 0 && !request_ptr->terminal()) {
                request_ptr->mark_failed("service stream dispatch made no progress");
            }
        }

        if (write_result == WriteResult::ok && writer.client_connected()) {
            const char* finish_event = request_ptr->timed_out()
                ? "data: {\"finish_reason\":\"timeout\"}\n\n"
                : request_ptr->cancelled()
                    ? "data: {\"finish_reason\":\"client_disconnect\"}\n\n"
                    : request_ptr->finished()
                        ? "data: {\"finish_reason\":\"stop\"}\n\n"
                        : "data: {\"finish_reason\":\"runtime_error\"}\n\n";
            write_result = writer.write_event(finish_event);
        }
        writer.finish();
    } catch (...) {
        adapter_exception = true;
        if (!request_ptr->terminal()) {
            request_ptr->mark_failed("service response writer threw an exception");
        }
        try {
            writer.finish();
        } catch (...) {
        }
    }

    const bool finished = request_ptr->finished();
    const bool cancelled = request_ptr->cancelled();
    const bool timed_out = request_ptr->timed_out();
    int status = 200;
    if (adapter_exception) status = 500;
    else if (write_result == WriteResult::backpressure) status = 503;
    else if (write_result == WriteResult::client_closed || cancelled) status = 499;
    else if (timed_out) status = 408;
    else if (!finished) status = 500;

    service_.release_terminal();
    requests_.erase(request_id);
    return {status, "text/event-stream", {}};
}

HttpResponse HttpAdapter::handle_metrics() const {
    const ServiceMetrics& metrics = service_.metrics();
    std::ostringstream body;
    body << "serving_accepted_total " << metrics.accepted_total << '\n'
         << "serving_queue_full_total " << metrics.queue_full_total << '\n'
         << "serving_duplicate_total " << metrics.duplicate_total << '\n'
         << "serving_completed_total " << metrics.completed_total << '\n'
         << "serving_cancelled_total " << metrics.cancelled_total << '\n'
         << "serving_timed_out_total " << metrics.timed_out_total << '\n'
         << "serving_failed_total " << metrics.failed_total << '\n'
         << "serving_released_total " << metrics.released_total << '\n'
         << "serving_generated_tokens_total " << metrics.generated_tokens_total << '\n'
         << "serving_inflight " << service_.inflight_size() << '\n'
         << "serving_queue_wait_seconds_sum " << metrics.queue_wait_seconds_sum << '\n'
         << "serving_queue_wait_seconds_count " << metrics.queue_wait_count << '\n'
         << "serving_ttft_seconds_sum " << metrics.ttft_seconds_sum << '\n'
         << "serving_ttft_seconds_count " << metrics.ttft_count << '\n'
         << "serving_completion_seconds_sum " << metrics.completion_seconds_sum << '\n'
         << "serving_completion_seconds_count " << metrics.completion_count << '\n';
    return {200, "text/plain; version=0.0.4", body.str()};
}

}  // namespace chapter10_serving
