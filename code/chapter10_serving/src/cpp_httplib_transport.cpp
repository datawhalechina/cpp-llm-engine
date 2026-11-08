// Chapter 10：实现 cpp-httplib 的 HTTP 与 chunked SSE 写回。
// 契约：网络库类型停留在本文件，并在进入单线程 ServiceCore 前串行化请求。
// 验证：可选 loopback 脚本检查 health、metrics、Completion 和 SSE。
#include "chapter10_serving/cpp_httplib_transport.h"

#include <httplib.h>

#include <mutex>
#include <utility>

namespace chapter10_serving {

namespace {

bool is_stream_request(const httplib::Request& request) {
    return request.body.find("\"stream\":true") != std::string::npos ||
           request.body.find("\"stream\": true") != std::string::npos;
}

HttpRequest to_http_request(const httplib::Request& request) {
    return {request.method, request.path, request.body};
}

void write_response(const HttpResponse& response, httplib::Response& target) {
    target.status = response.status_code;
    target.set_content(response.body, response.content_type);
}

class HttplibResponseWriter final : public ResponseWriter {
public:
    explicit HttplibResponseWriter(httplib::DataSink& sink) : sink_(sink) {}

    bool write_headers(const ResponseHead&) override { return connected_; }

    WriteResult write_event(std::string_view event) override {
        if (!connected_) return WriteResult::client_closed;
        if (!sink_.write(event.data(), event.size())) {
            connected_ = false;
            return WriteResult::client_closed;
        }
        return WriteResult::ok;
    }

    void finish() override {
        if (finished_) return;
        sink_.done();
        finished_ = true;
    }

    bool client_connected() const override { return connected_; }

private:
    httplib::DataSink& sink_;
    bool connected_ = true;
    bool finished_ = false;
};

}  // namespace

CppHttplibServer::CppHttplibServer(
    HttpAdapter& adapter,
    std::string bind_address,
    int port)
    : adapter_(adapter), bind_address_(std::move(bind_address)), port_(port) {}

int CppHttplibServer::listen() {
    httplib::Server server;
    std::mutex adapter_mutex;
    server.Get("/health", [&](const httplib::Request& request, httplib::Response& response) {
        const std::lock_guard<std::mutex> lock(adapter_mutex);
        write_response(adapter_.handle(to_http_request(request)), response);
    });
    server.Get("/metrics", [&](const httplib::Request& request, httplib::Response& response) {
        const std::lock_guard<std::mutex> lock(adapter_mutex);
        write_response(adapter_.handle(to_http_request(request)), response);
    });
    server.Post(
        "/v1/completions",
        [&](const httplib::Request& request, httplib::Response& response) {
            const HttpRequest http_request = to_http_request(request);
            if (!is_stream_request(request)) {
                const std::lock_guard<std::mutex> lock(adapter_mutex);
                write_response(adapter_.handle(http_request), response);
                return;
            }
            response.set_chunked_content_provider(
                "text/event-stream",
                [this, &adapter_mutex, http_request](
                    std::size_t,
                    httplib::DataSink& sink) mutable {
                    const std::lock_guard<std::mutex> lock(adapter_mutex);
                    HttplibResponseWriter writer(sink);
                    (void)adapter_.handle_stream(http_request, writer);
                    writer.finish();
                    return true;
                });
        });
    return server.listen(bind_address_, port_) ? 0 : 1;
}

}  // namespace chapter10_serving
