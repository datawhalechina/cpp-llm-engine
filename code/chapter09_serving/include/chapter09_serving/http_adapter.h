// Chapter 09：声明不绑定网络库的 HTTP request/response adapter。
// 职责：把健康检查、指标和 completion 请求映射到 ServingCore 与 Runtime。
// 验证：HTTP adapter 测试覆盖成功响应、输入错误、404、容量错误和 metrics 输出。
#pragma once

#include "chapter09_serving/serving_core.h"
#include "chapter09_serving/response_writer.h"

#include <cstddef>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>

namespace chapter09_serving {

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
        ServingCore& service,
        chapter06_generation::Runtime& runtime,
        const chapter06_generation::Sampler& sampler,
        RequestExecutorFactory executor_factory = {});

    HttpResponse handle(const HttpRequest& request);
    HttpResponse handle_stream(const HttpRequest& request, ResponseWriter& writer);

private:
    HttpResponse handle_completion(const HttpRequest& request);
    HttpResponse handle_stream_completion(const HttpRequest& request, ResponseWriter& writer);
    HttpResponse handle_metrics() const;

    ServingCore& service_;
    chapter06_generation::Runtime& runtime_;
    const chapter06_generation::Sampler& sampler_;
    RequestExecutorFactory executor_factory_;
    std::unordered_map<std::size_t, std::unique_ptr<chapter06_generation::Request>> requests_;
};

}  // namespace chapter09_serving
