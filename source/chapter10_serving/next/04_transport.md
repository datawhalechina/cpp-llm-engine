# 10.4 让 Transport 只处理连接和字节

Transport 负责监听、HTTP 解析、响应写出和连接事件；<a href="../../_code/chapter10_serving/include/chapter10_serving/http_adapter.h"><code>HttpAdapter</code></a> 负责路由与参数语义；`ServiceCore` 负责请求生命周期。更换网络库时，后两层契约保持不变。

## 三层映射

| 层 | 输入与输出 | 不应进入 |
| --- | --- | --- |
| Transport | socket、HTTP method/path/body、chunk | Request 状态和调度策略 |
| HttpAdapter | Completion 参数、状态码、SSE event | 网络库类型和模型 Kernel |
| ServiceCore | admission、dispatch、cancel、metrics | HTTP header 和连接对象 |

## 网络库选择

| 方案 | 教程中的位置 | 主要取舍 |
| --- | --- | --- |
| cpp-httplib | 首个真实 HTTP/SSE 回环 | 接入短，适合验证同步 HTTP/1.1 |
| Boost.Asio/Beast | 异步连接、Deadline 和背压扩展 | 生命周期完整，状态机更复杂 |
| Crow/oat++ | Web 框架对照 | 路由丰富，但容易遮蔽核心契约 |

`ResponseWriter` 是 Transport 与 Adapter 之间的窄接口，只暴露 headers、event、finish 和连接状态。网络库对象不应穿过该接口进入 ServiceCore。

服务进程的模型路径、监听地址、端口、后端和实现策略适合由命令行或部署环境提供。Transport 在启动时完成解析，再把类型化配置交给 Engine Factory 和 `ServiceCore`；请求处理期间不再读取 `argv`。推荐保持如下优先级：显式命令行参数覆盖环境默认值，环境默认值覆盖程序内安全默认值，并在启动日志打印最终配置。请求自身的生成参数仍由 `HttpAdapter` 校验，不能与进程级 backend 配置混用。

cpp-httplib 可以并发接收连接，但当前 `ServiceCore` 是单线程状态机。`CppHttplibServer` 因此在调用 `HttpAdapter` 前持有同一把互斥锁；同步 Completion 或 SSE 生命周期结束后，下一个请求才进入 Adapter。这个边界先保证状态和资源安全，不代表已经实现多个 HTTP 请求共同进入 StepBatchPlan。后续异步 Transport 需要把连接事件与 ServiceCore 的单线程事件循环分开，而不是简单移除互斥锁。

## 验证 Transport 可替换性

先运行 `bash code/chapter10_serving/scripts/run.sh`，确认 `chapter10_http_contract_test` 在不启动 socket 的情况下固定 Adapter/Writer 契约。本地已有 cpp-httplib 时，再运行 `bash code/chapter10_serving/scripts/run_cpp_httplib_loopback.sh`，通过条件是 health、metrics、同步 Completion 和 SSE 回环全部成功，且所有 Adapter 调用经过串行入口。`10.5` 继续追踪一次同步 Completion。
