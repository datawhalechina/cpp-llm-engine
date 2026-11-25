// Chapter 10: map HTTP/SSE to synchronous or background-owned ServiceCore execution.
// The asynchronous path owns Request lifetime in AsyncServiceLoop.
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

bool read_string(
    const std::string& body,
    const std::string& key,
    std::string* value) {
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
    if (position >= body.size() || body[position] != '"') return false;
    ++position;
    std::string parsed;
    while (position < body.size()) {
        const char current = body[position++];
        if (current == '"') {
            *value = std::move(parsed);
            return true;
        }
        if (current != '\\') {
            parsed.push_back(current);
            continue;
        }
        if (position >= body.size()) return false;
        const char escaped = body[position++];
        switch (escaped) {
            case '"': parsed.push_back('"'); break;
            case '\\': parsed.push_back('\\'); break;
            case 'n': parsed.push_back('\n'); break;
            case 'r': parsed.push_back('\r'); break;
            case 't': parsed.push_back('\t'); break;
            default: return false;
        }
    }
    return false;
}

bool read_prompt(
    const std::string& body,
    const PromptTokenizer& tokenizer,
    chapter06_generation::TokenSequence* tokens) {
    if (read_tokens(body, tokens)) return true;
    if (!tokenizer) return false;
    std::string prompt;
    if (!read_string(body, "prompt", &prompt) || prompt.empty()) return false;
    try {
        *tokens = tokenizer(prompt);
    } catch (...) {
        tokens->clear();
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
    RequestExecutorFactory executor_factory,
    PromptTokenizer prompt_tokenizer,
    CompletionRequestPreparer completion_preparer,
    AsyncServiceLoop* async_service,
    CompletionRequestCollector* preparation_collector)
    : service_(service),
      runtime_(runtime),
      sampler_(sampler),
      executor_factory_(std::move(executor_factory)),
      prompt_tokenizer_(std::move(prompt_tokenizer)),
      completion_preparer_(std::move(completion_preparer)),
      async_service_(async_service),
      preparation_collector_(preparation_collector) {}

HttpAdapter::~HttpAdapter() = default;

bool HttpAdapter::asynchronous() const { return async_service_ != nullptr; }

CompletionPrepareStatus HttpAdapter::prepare_completion(
    const HttpRequest& request,
    PreparedCompletionRequest* prepared,
    std::string* error) const {
    if (prepared == nullptr) {
        if (error != nullptr) *error = "prepared completion output must not be null";
        return CompletionPrepareStatus::invalid_request;
    }
    *prepared = {};
    if (preparation_collector_ != nullptr) {
        CompletionPreparationResult result =
            preparation_collector_->prepare(request);
        *prepared = std::move(result.prepared);
        if (error != nullptr) *error = std::move(result.error);
        return result.status;
    }
    if (completion_preparer_) {
        const auto status = completion_preparer_(request, prepared, error);
        if (status != CompletionPrepareStatus::ready) return status;
        if (prepared->prompt_positions.empty()) {
            if (error != nullptr) *error = "prepared completion has no prompt positions";
            return CompletionPrepareStatus::invalid_request;
        }
        if (prepared->executor == nullptr) {
            if (error != nullptr) *error = "prepared completion has no executor";
            return CompletionPrepareStatus::executor_unavailable;
        }
        return CompletionPrepareStatus::ready;
    }

    if (!read_prompt(request.body, prompt_tokenizer_, &prepared->prompt_positions)) {
        if (error != nullptr) *error = "completion prompt is invalid";
        return CompletionPrepareStatus::invalid_request;
    }
    if (executor_factory_) {
        prepared->executor = executor_factory_();
        if (prepared->executor == nullptr) {
            if (error != nullptr) *error = "model executor is unavailable";
            return CompletionPrepareStatus::executor_unavailable;
        }
    }
    if (error != nullptr) error->clear();
    return CompletionPrepareStatus::ready;
}

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
    if (!read_number(request.body, "request_id", &request_id)) {
        return json_response(400, "{\"error\":\"invalid_completion_body\"}");
    }
    PreparedCompletionRequest prepared;
    std::string prepare_error;
    const auto prepare_status = prepare_completion(
        request, &prepared, &prepare_error);
    if (prepare_status == CompletionPrepareStatus::invalid_request) {
        return json_response(400, "{\"error\":\"invalid_completion_body\"}");
    }
    if (prepare_status == CompletionPrepareStatus::executor_unavailable) {
        return json_response(500, "{\"error\":\"model_executor_unavailable\"}");
    }
    if (async_service_ != nullptr) {
        return handle_async_completion(request, request_id, std::move(prepared));
    }
    auto owned_request = std::make_unique<chapter06_generation::Request>(
        std::move(prepared.prompt_positions));
    const AdmissionResult admission = service_.admit(
        request_id, *owned_request, sampler_, std::move(prepared.executor));
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
    if (!read_number(request.body, "request_id", &request_id)) {
        return json_response(400, "{\"error\":\"invalid_completion_body\"}");
    }
    PreparedCompletionRequest prepared;
    std::string prepare_error;
    const auto prepare_status = prepare_completion(
        request, &prepared, &prepare_error);
    if (prepare_status == CompletionPrepareStatus::invalid_request) {
        return json_response(400, "{\"error\":\"invalid_completion_body\"}");
    }
    if (prepare_status == CompletionPrepareStatus::executor_unavailable) {
        return json_response(500, "{\"error\":\"model_executor_unavailable\"}");
    }
    if (async_service_ != nullptr) {
        return handle_async_stream(
            request, request_id, std::move(prepared), writer);
    }
    auto owned_request = std::make_unique<chapter06_generation::Request>(
        std::move(prepared.prompt_positions));
    const AdmissionResult admission = service_.admit(
        request_id, *owned_request, sampler_, std::move(prepared.executor));
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

HttpResponse HttpAdapter::handle_async_completion(
    const HttpRequest& request,
    std::size_t request_id,
    PreparedCompletionRequest prepared) {
    const AsyncRequestHandle handle = async_service_->submit(
        request_id, std::move(prepared.prompt_positions),
        std::move(prepared.executor));
    const AdmissionResult admission = handle.wait_admission();
    if (!admission.accepted()) return admission_error(admission.status);
    if (!request.client_connected) async_service_->cancel(request_id);

    std::size_t cursor = 0;
    AsyncTokenEvent event;
    while (handle.wait_next(&cursor, &event)) {
        if (request.disconnect_after_tokens > 0 &&
            cursor >= request.disconnect_after_tokens) {
            async_service_->cancel(request_id);
        } else if (request.timeout_after_tokens > 0 &&
                   cursor >= request.timeout_after_tokens) {
            async_service_->timeout(request_id);
        }
    }
    const AsyncRequestResult result = handle.wait();
    if (result.status == AsyncCompletionStatus::rejected) {
        return admission_error(result.admission);
    }
    if (result.status == AsyncCompletionStatus::cancelled) {
        return json_response(499, "{\"error\":\"client_disconnect\"}");
    }
    if (result.status == AsyncCompletionStatus::timed_out) {
        return json_response(408, "{\"error\":\"timeout\"}");
    }
    if (result.status == AsyncCompletionStatus::stopped) {
        return json_response(503, "{\"error\":\"service_stopped\"}");
    }
    if (result.status != AsyncCompletionStatus::completed) {
        return json_response(500, "{\"error\":\"runtime_did_not_finish\"}");
    }

    std::ostringstream body;
    body << "{\"request_id\":" << request_id << ",\"generated_token_ids\":[";
    for (std::size_t index = 0; index < result.generated.size(); ++index) {
        if (index > 0) body << ',';
        body << result.generated[index];
    }
    body << "],\"finish_reason\":\"stop\"}";
    return json_response(200, body.str());
}

HttpResponse HttpAdapter::handle_async_stream(
    const HttpRequest& request,
    std::size_t request_id,
    PreparedCompletionRequest prepared,
    ResponseWriter& writer) {
    const AsyncRequestHandle handle = async_service_->submit(
        request_id, std::move(prepared.prompt_positions),
        std::move(prepared.executor));
    const AdmissionResult admission = handle.wait_admission();
    if (!admission.accepted()) return admission_error(admission.status);
    WriteResult write_result = WriteResult::ok;
    bool adapter_exception = false;
    try {
        if (!request.client_connected) {
            write_result = WriteResult::client_closed;
            async_service_->cancel(request_id);
        } else if (!writer.write_headers({200, "text/event-stream"})) {
            write_result = writer.client_connected()
                ? WriteResult::backpressure : WriteResult::client_closed;
            async_service_->cancel(request_id);
        }

        std::size_t cursor = 0;
        AsyncTokenEvent token_event;
        while (handle.wait_next(&cursor, &token_event)) {
            if (write_result != WriteResult::ok) continue;
            std::ostringstream event;
            event << "data: {\"token\":" << token_event.token
                  << ",\"step\":" << token_event.step << "}\n\n";
            write_result = writer.write_event(event.str());
            if (write_result != WriteResult::ok) {
                async_service_->cancel(request_id);
            } else if (request.disconnect_after_tokens > 0 &&
                       cursor >= request.disconnect_after_tokens) {
                async_service_->cancel(request_id);
            } else if (request.timeout_after_tokens > 0 &&
                       cursor >= request.timeout_after_tokens) {
                async_service_->timeout(request_id);
            }
        }

        const AsyncRequestResult result = handle.wait();
        if (write_result == WriteResult::ok && writer.client_connected()) {
            const char* finish_event = result.status == AsyncCompletionStatus::timed_out
                ? "data: {\"finish_reason\":\"timeout\"}\n\n"
                : result.status == AsyncCompletionStatus::cancelled
                    ? "data: {\"finish_reason\":\"client_disconnect\"}\n\n"
                    : result.status == AsyncCompletionStatus::completed
                        ? "data: {\"finish_reason\":\"stop\"}\n\n"
                        : "data: {\"finish_reason\":\"runtime_error\"}\n\n";
            write_result = writer.write_event(finish_event);
        }
        writer.finish();

        if (result.status == AsyncCompletionStatus::rejected) {
            return admission_error(result.admission);
        }
        if (write_result == WriteResult::backpressure) {
            return {503, "text/event-stream", {}};
        }
        if (write_result == WriteResult::client_closed ||
            result.status == AsyncCompletionStatus::cancelled) {
            return {499, "text/event-stream", {}};
        }
        if (result.status == AsyncCompletionStatus::timed_out) {
            return {408, "text/event-stream", {}};
        }
        if (result.status != AsyncCompletionStatus::completed) {
            return {500, "text/event-stream", {}};
        }
        return {200, "text/event-stream", {}};
    } catch (...) {
        adapter_exception = true;
        async_service_->cancel(request_id);
        try {
            (void)handle.wait();
            writer.finish();
        } catch (...) {
        }
    }
    return {adapter_exception ? 500 : 200, "text/event-stream", {}};
}

HttpResponse HttpAdapter::handle_metrics() const {
    const AsyncServiceSnapshot async_snapshot = async_service_ == nullptr
        ? AsyncServiceSnapshot{}
        : async_service_->snapshot();
    const ServiceMetrics& metrics = async_service_ == nullptr
        ? service_.metrics()
        : async_snapshot.metrics;
    const std::size_t waiting = async_service_ == nullptr
        ? service_.waiting_size() : async_snapshot.waiting;
    const std::size_t running = async_service_ == nullptr
        ? service_.running_size() : async_snapshot.running;
    std::ostringstream body;
    body << "serving_accepted_total " << metrics.accepted_total << '\n'
         << "serving_queue_full_total " << metrics.queue_full_total << '\n'
         << "serving_duplicate_total " << metrics.duplicate_total << '\n'
         << "serving_invalid_total " << metrics.invalid_total << '\n'
         << "serving_completed_total " << metrics.completed_total << '\n'
         << "serving_cancelled_total " << metrics.cancelled_total << '\n'
         << "serving_timed_out_total " << metrics.timed_out_total << '\n'
         << "serving_failed_total " << metrics.failed_total << '\n'
         << "serving_released_total " << metrics.released_total << '\n'
         << "serving_generated_tokens_total " << metrics.generated_tokens_total << '\n'
         << "serving_inflight " << waiting + running << '\n'
         << "serving_waiting " << waiting << '\n'
         << "serving_running " << running << '\n'
         << "serving_peak_inflight " << metrics.peak_inflight << '\n'
         << "serving_queue_wait_seconds_sum " << metrics.queue_wait_seconds_sum << '\n'
         << "serving_queue_wait_seconds_count " << metrics.queue_wait_count << '\n'
         << "serving_ttft_seconds_sum " << metrics.ttft_seconds_sum << '\n'
         << "serving_ttft_seconds_count " << metrics.ttft_count << '\n'
         << "serving_completion_seconds_sum " << metrics.completion_seconds_sum << '\n'
         << "serving_completion_seconds_count " << metrics.completion_count << '\n';
    return {200, "text/plain; version=0.0.4", body.str()};
}

}  // namespace chapter10_serving
