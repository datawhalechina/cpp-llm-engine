// Chapter 10：声明不绑定网络库的最小 HTTP Completion 适配器。
// 职责：把路由、请求字段和 SSE 事件映射到 ServiceCore，不创建 socket。
// 验证：chapter10_http_contract_test 覆盖路由、状态码、流式终态和指标。
#pragma once

#include "chapter10_serving/response_writer.h"
#include "chapter10_serving/service_core.h"

#include <functional>
#include <memory>
#include <string>
#include <unordered_map>

namespace chapter10_serving {

using RequestExecutorFactory = std::function<
    std::shared_ptr<chapter06_generation::RequestModelExecutor>()>;

struct HttpRequest {
    std::string method;
    std::string path;
    std::string body;
    bool client_connected = true;
    std::size_t disconnect_after_tokens = 0;
    std::size_t timeout_after_tokens = 0;
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
        RequestExecutorFactory executor_factory = {});
    ~HttpAdapter();

    HttpResponse handle(const HttpRequest& request);
    HttpResponse handle_stream(const HttpRequest& request, ResponseWriter& writer);

private:
    HttpResponse handle_completion(const HttpRequest& request);
    HttpResponse handle_metrics() const;

    ServiceCore& service_;
    chapter06_generation::Runtime& runtime_;
    const chapter06_generation::Sampler& sampler_;
    RequestExecutorFactory executor_factory_;
    std::unordered_map<std::size_t, std::unique_ptr<chapter06_generation::Request>> requests_;
};

}  // namespace chapter10_serving
