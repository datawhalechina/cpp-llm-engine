// Chapter 10：声明 cpp-httplib 到 Transport-neutral HttpAdapter 的适配器。
// 职责：接收真实 HTTP 请求并写回同步或 SSE 响应，不保存服务状态。
// 验证：run_cpp_httplib_loopback.sh 检查本地 socket 闭环。
#pragma once

#include "chapter10_serving/http_adapter.h"

#include <string>

namespace chapter10_serving {

class CppHttplibServer final {
public:
    CppHttplibServer(HttpAdapter& adapter, std::string bind_address, int port);
    int listen();

private:
    HttpAdapter& adapter_;
    std::string bind_address_;
    int port_;
};

}  // namespace chapter10_serving
