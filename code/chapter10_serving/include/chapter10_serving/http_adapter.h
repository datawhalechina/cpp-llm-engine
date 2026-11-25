// Chapter 10: declare a transport-neutral HTTP Completion adapter.
// It maps routes and SSE events to synchronous or asynchronous ServiceCore execution.
// 验证：chapter10_http_contract_test 覆盖路由、状态码、流式终态和指标。
#pragma once

#include "chapter10_serving/async_service.h"
#include "chapter10_serving/response_writer.h"
#include "chapter10_serving/service_core.h"

#include <functional>
#include <chrono>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace chapter10_serving {

struct HttpRequest {
    std::string method;
    std::string path;
    std::string body;
    bool client_connected = true;
    std::size_t disconnect_after_tokens = 0;
    std::size_t timeout_after_tokens = 0;
};

using RequestExecutorFactory = std::function<
    std::shared_ptr<chapter06_generation::RequestModelExecutor>()>;
using PromptTokenizer = std::function<chapter06_generation::TokenSequence(
    const std::string&)>;

struct PreparedCompletionRequest {
    chapter06_generation::TokenSequence prompt_positions;
    std::shared_ptr<chapter06_generation::RequestModelExecutor> executor;
};

enum class CompletionPrepareStatus {
    ready,
    invalid_request,
    executor_unavailable,
};

using CompletionRequestPreparer = std::function<CompletionPrepareStatus(
    const HttpRequest&, PreparedCompletionRequest*, std::string*)>;

struct CompletionPreparationResult {
    CompletionPrepareStatus status = CompletionPrepareStatus::invalid_request;
    PreparedCompletionRequest prepared;
    std::string error;
};

using CompletionBatchPreparer = std::function<void(
    const std::vector<HttpRequest>&,
    std::vector<CompletionPreparationResult>*)>;

struct CompletionCollectorStats {
    std::size_t batch_calls = 0;
    std::size_t prepared_requests = 0;
    std::size_t max_batch_size = 0;
};

class CompletionRequestCollector final {
public:
    CompletionRequestCollector(
        CompletionBatchPreparer preparer,
        std::size_t max_batch_size,
        std::chrono::microseconds collect_window =
            std::chrono::microseconds(1000));
    ~CompletionRequestCollector();

    CompletionRequestCollector(const CompletionRequestCollector&) = delete;
    CompletionRequestCollector& operator=(const CompletionRequestCollector&) = delete;

    CompletionPreparationResult prepare(const HttpRequest& request);
    CompletionCollectorStats stats() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

struct HttpResponse {
    int status_code = 500;
    std::string content_type = "application/json";
    std::string body;
};

class HttpAdapter final {
public:
    HttpAdapter(
        ServiceCore& service,
        chapter06_generation::Runtime& runtime,
        const chapter06_generation::Sampler& sampler,
        RequestExecutorFactory executor_factory = {},
        PromptTokenizer prompt_tokenizer = {},
        CompletionRequestPreparer completion_preparer = {},
        AsyncServiceLoop* async_service = nullptr,
        CompletionRequestCollector* preparation_collector = nullptr);
    ~HttpAdapter();

    HttpResponse handle(const HttpRequest& request);
    HttpResponse handle_stream(const HttpRequest& request, ResponseWriter& writer);
    bool asynchronous() const;

private:
    HttpResponse handle_completion(const HttpRequest& request);
    HttpResponse handle_metrics() const;
    CompletionPrepareStatus prepare_completion(
        const HttpRequest& request,
        PreparedCompletionRequest* prepared,
        std::string* error) const;
    HttpResponse handle_async_completion(
        const HttpRequest& request,
        std::size_t request_id,
        PreparedCompletionRequest prepared);
    HttpResponse handle_async_stream(
        const HttpRequest& request,
        std::size_t request_id,
        PreparedCompletionRequest prepared,
        ResponseWriter& writer);

    ServiceCore& service_;
    chapter06_generation::Runtime& runtime_;
    const chapter06_generation::Sampler& sampler_;
    RequestExecutorFactory executor_factory_;
    PromptTokenizer prompt_tokenizer_;
    CompletionRequestPreparer completion_preparer_;
    AsyncServiceLoop* async_service_ = nullptr;
    CompletionRequestCollector* preparation_collector_ = nullptr;
    std::unordered_map<std::size_t, std::unique_ptr<chapter06_generation::Request>> requests_;
};

}  // namespace chapter10_serving
