# chapter10_serving 代码导航

本章代码把网络映射、服务生命周期和调度执行分开。cpp-httplib 可以接收多个连接，但当前 Transport 会串行调用单线程 `HttpAdapter`/`ServiceCore`；模型侧多请求执行由 Chapter08 公共接口负责。

## 阅读顺序

1. 先看 `service_core.h/.cpp`，跟踪 admission、waiting/running 和统一释放
2. 再看 `response_writer.h`，确认 Transport 与服务核心之间的最窄接口
3. 最后看 `http_adapter.h/.cpp`、Transport 和四组测试，核对状态码、SSE、并发边界与资源归零

## 服务核心

| 文件 | 作用 |
| --- | --- |
| <a href="https://github.com/datawhalechina/cpp-llm-engine/blob/main/code/chapter10_serving/include/chapter10_serving/service_core.h"><code>service_core.h</code></a> | 服务状态、准入结果、dispatch 与 metrics 接口 |
| <a href="https://github.com/datawhalechina/cpp-llm-engine/blob/main/code/chapter10_serving/src/service_core.cpp"><code>service_core.cpp</code></a> | Chapter08 调度交接和统一终态释放 |
| <a href="https://github.com/datawhalechina/cpp-llm-engine/blob/main/code/chapter10_serving/include/chapter10_serving/request_worker.h"><code>request_worker.h</code></a> | 请求级 CPU Worker 与执行结果契约 |
| <a href="https://github.com/datawhalechina/cpp-llm-engine/blob/main/code/chapter10_serving/src/request_worker.cpp"><code>request_worker.cpp</code></a> | 独立 Request/Executor 的并行推进 |
| <a href="https://github.com/datawhalechina/cpp-llm-engine/blob/main/code/chapter10_serving/tests/service_core_test.cpp"><code>service_core_test.cpp</code></a> | 容量、waiting/running 和 step 调度 |
| <a href="https://github.com/datawhalechina/cpp-llm-engine/blob/main/code/chapter10_serving/tests/terminal_release_test.cpp"><code>terminal_release_test.cpp</code></a> | 取消、超时、完成、Executor 失败和 K/V 容量恢复 |

## HTTP 与流式输出

| 文件 | 作用 |
| --- | --- |
| <a href="https://github.com/datawhalechina/cpp-llm-engine/blob/main/code/chapter10_serving/include/chapter10_serving/response_writer.h"><code>response_writer.h</code></a> | headers、event、连接和背压契约 |
| <a href="https://github.com/datawhalechina/cpp-llm-engine/blob/main/code/chapter10_serving/include/chapter10_serving/http_adapter.h"><code>http_adapter.h</code></a> | HTTP request/response 与 Factory 接口 |
| <a href="https://github.com/datawhalechina/cpp-llm-engine/blob/main/code/chapter10_serving/src/http_adapter.cpp"><code>http_adapter.cpp</code></a> | Completion、SSE、metrics 和错误映射 |
| <a href="https://github.com/datawhalechina/cpp-llm-engine/blob/main/code/chapter10_serving/src/cpp_httplib_transport.cpp"><code>cpp_httplib_transport.cpp</code></a> | 可选 cpp-httplib socket 适配与 Adapter 串行入口 |
| <a href="https://github.com/datawhalechina/cpp-llm-engine/blob/main/code/chapter10_serving/tests/http_contract_test.cpp"><code>http_contract_test.cpp</code></a> | 同步响应、SSE、断连、超时、背压和资源释放 |
| <a href="https://github.com/datawhalechina/cpp-llm-engine/blob/main/code/chapter10_serving/tests/request_worker_test.cpp"><code>request_worker_test.cpp</code></a> | 请求级执行重叠与输出隔离 |
| <a href="https://github.com/datawhalechina/cpp-llm-engine/blob/main/code/chapter10_serving/scripts/run.sh"><code>scripts/run.sh</code></a> | Chapter10 独立构建与 CTest 入口 |
| <a href="https://github.com/datawhalechina/cpp-llm-engine/blob/main/code/chapter10_serving/scripts/run_cpp_httplib_loopback.sh"><code>run_cpp_httplib_loopback.sh</code></a> | 可选真实 HTTP/SSE 回环 |
