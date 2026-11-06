// Chapter 10：声明不绑定网络库的流式响应写出接口。
// 职责：传递 headers、SSE event、连接关闭和背压，不持有模型状态。
// 验证：chapter10_http_contract_test 覆盖正常、断连和背压路径。
#pragma once

#include <string>
#include <string_view>

namespace chapter10_serving {

struct ResponseHead {
    int status_code = 500;
    std::string content_type = "application/json";
};

enum class WriteResult {
    ok,
    client_closed,
    backpressure,
};

class ResponseWriter {
public:
    virtual ~ResponseWriter() = default;
    virtual bool write_headers(const ResponseHead& head) = 0;
    virtual WriteResult write_event(std::string_view event) = 0;
    virtual void finish() = 0;
    virtual bool client_connected() const = 0;
};

}  // namespace chapter10_serving
