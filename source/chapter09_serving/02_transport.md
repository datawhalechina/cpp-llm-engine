# 09.2 选择 Transport 与 C++ 网络库

网络库负责连接、HTTP 解析、响应写出和连接事件；`HttpAdapter` 与 `ServiceCore` 负责参数语义、请求状态、token event、指标和资源回收。替换 transport 时，这些服务契约不能随网络库重写。

## Transport 怎样接入服务核心

同步测试使用 `HttpAdapter::handle()` 返回完整响应，流式路径使用 `handle_stream()` 和 `ResponseWriter` 逐个写出事件。网络库类型停留在最外层适配器，不进入 ServiceCore。

```text
cpp-httplib / Boost.Beast
            |
      Transport Adapter
            |
       ResponseWriter
            |
        HttpAdapter
            |
 ServiceCore -> Runtime -> Executor
```

| 事件 | Transport 负责 | ServiceCore 负责 |
| --- | --- | --- |
| 请求到达 | method、path、body 解析 | 参数校验与 admission |
| 正常输出 | headers、SSE event、finish | token 顺序与请求终态 |
| 客户端断开 | 报告连接关闭 | 取消请求并释放资源 |
| Deadline | 触发超时事件 | 停止调度并进入 `TimedOut` |
| 背压 | 报告暂时无法写出 | 暂停、取消或记录失败 |

## 网络库与 Writer 契约

第一阶段使用 `cpp-httplib` 验证真实 HTTP/SSE 回环，因为它接入简单且不改变核心接口。需要异步连接生命周期、超时、背压和 TLS 时，再用 Boost.Asio/Beast 实现同一个 adapter；不应为了更换网络库再写一套调度器。

`ResponseWriter` 只提供 `write_headers`、`write_event`、`finish` 和 `client_connected`。`WriteResult` 区分成功、客户端关闭和背压。HTTP headers 已发送后不能再修改状态码，因此断开或超时通过最终 event 表达，同时让 ServiceCore 记录 `499/client_disconnect` 或 `408/timeout` 语义终态。

| 方案 | 适合的位置 | 主要取舍 |
| --- | --- | --- |
| `cpp-httplib` | 首个真实 HTTP/SSE 闭环 | 单头文件、易验证；以阻塞式 HTTP/1.1 为主 |
| Boost.Asio + Beast | 异步 transport 扩展 | 生命周期与背压能力完整；状态机和类型更多 |
| Crow / oat++ | 完整 Web 框架对照 | 路由与组件丰富；容易遮蔽本教程关注的适配边界 |

## 验证真实回环

默认入口验证 transport-neutral 的同步、SSE、断开、超时、背压和资源回收。准备本地 `httplib.h` 后，可额外运行真实 socket 回环：

```bash
CHAPTER09_CPPHTTPLIB_INCLUDE_DIR=/path/to/cpp-httplib \
bash code/chapter09_serving/scripts/run_cpp_httplib_loopback.sh
```

未设置 include path 时，CMake 会尝试获取固定版本 `v0.18.3`。测试依次请求 `/health`、`/metrics`、同步 completion 和 SSE completion；真实 TinyLlama HTTP 综合检查由 `bash code/engine/scripts/run_real_http_server.sh --check` 执行。

| 场景 | 必须观察的结果 |
| --- | --- |
| `/health`、`/metrics` | HTTP 200，ServiceCore 指标语义不变 |
| 同步 completion | status、body 和 finish reason 正确 |
| SSE completion | token 顺序和最终 finish event 正确 |
| 客户端断开 | `499`、请求终止、cache 释放 |
| Deadline | `408`、停止后续 decode、cache 释放 |
| 背压 | 不重复 token，不绕过请求终态 |

## 验证与衔接

读完后应能解释为什么 transport 可以替换，而 Request、metrics 和 streaming contract 必须保持稳定。下一节 [`09.3 真实 ServiceCore`](./03_real_service_benchmark.md) 测量 transport 以内的模型与运行时成本；真实网络延迟需要单独设计 HTTP workload。
