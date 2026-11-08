# Chapter 10 代码：推理服务

本章把 Chapter06 请求状态、Chapter07 KV 资源和 Chapter08 调度器接入同一个服务生命周期，并通过与网络库无关的接口返回同步或流式结果。进入源码前先阅读 [`Chapter10 导读`](../../source/chapter10_serving/next/intro.md)。

## 代码架构

`CppHttplibServer` 只处理 socket 和 chunked write，并在进入当前单线程 `HttpAdapter` 前串行化调用。Adapter 解析 Completion 参数、持有请求对象并映射 HTTP/SSE 结果；`ServiceCore` 管理 waiting/running、准入、指标和终态释放，再把每个执行步交给 Chapter08。模型状态由请求专属 Executor/Context 持有，分页 K/V 通过 Chapter07 事务释放。

| 调用阶段 | 主要对象 | 所有权与失败处理 |
| --- | --- | --- |
| 网络接入 | `CppHttplibServer`、`ResponseWriter` | Transport 拥有连接；断连或写回失败转换为取消 |
| 协议映射 | `HttpAdapter` | Adapter 持有 Request，终态释放后才销毁 |
| 生命周期 | `ServiceCore` | 保存 waiting/running 引用，统一统计并释放请求资源 |
| 调度执行 | Chapter08 Scheduler、Executor | 每步提交候选；失败写入 Request 终态 |
| 缓存回收 | Chapter07 cache transaction | 完成、取消、超时和失败共用释放入口 |

当前互斥入口保证 Transport 不会并发修改 `ServiceCore`，但也使真实 HTTP Completion/SSE 串行执行。`RequestWorkerPool` 是独立 Context 的 CPU 重叠实验；Chapter08 Tensor Batch 是一次模型调用处理多请求，两者不属于当前 HTTP Transport 的并发能力。

## 阅读顺序与资产

| 模块 | 学习内容 | 验证 |
| --- | --- | --- |
| `ServiceCore` | 准入、等待/运行队列、调度交接、指标和统一释放 | `chapter10_service_core_test` |
| `RequestWorkerPool` | 用独立 Request 与 Executor 并行执行 CPU 请求 | `chapter10_request_worker_test` |
| `ResponseWriter` | 表达响应头、SSE 事件、断连和背压 | `chapter10_http_contract_test` |
| `HttpAdapter` | 把 health、metrics 和 Completion 路由映射到 ServiceCore | `chapter10_http_contract_test` |
| `CppHttplibServer` | 把可选 cpp-httplib socket 接到 HttpAdapter | `run_cpp_httplib_loopback.sh` |
| 终态释放 | 从 waiting/running 移除请求并释放 K/V 与批执行状态 | `chapter10_terminal_release_test` |

1. 阅读 `ServiceCore::admit()` 和 `promote_waiting()`，理解服务容量与调度预算的区别。
2. 阅读 `dispatch_step()`，观察服务如何把运行请求交给 Chapter08，而不是重新实现调度。
3. 阅读 `release_terminal()`，确认完成、取消、超时和失败共用一条资源释放路径。
4. 阅读 `HttpAdapter` 与 `ResponseWriter`，跟踪同步 Completion 和 SSE 如何处理断连、背压与异常。
5. 最后阅读 `CppHttplibServer`，查看网络库如何只承担 Transport 适配。

请求从 waiting 进入 running 后，由 Chapter08 每步选择执行对象。请求到达终态并不等于资源已经释放；调用 `release_terminal()` 后，Request 缓存、批执行绑定和分页 KV 块才会一起回收。

## 运行与验证

默认脚本不访问网络，运行服务状态、HTTP 契约、并行 Worker 和终态释放测试：

```bash
bash code/chapter10_serving/scripts/run.sh
```

可选 socket 闭环使用：

```bash
bash code/chapter10_serving/scripts/run_cpp_httplib_loopback.sh
```

准备本地 `httplib.h` 后可以设置 `CHAPTER10_CPPHTTPLIB_INCLUDE_DIR`；未设置时，CMake 获取固定版本 `v0.18.3`。该入口验证 health、metrics、同步 Completion 和 SSE，不代表并发 HTTP 服务。

## 扩展入口

ServiceCore 已接通 Chapter08 优先级策略、真实 Decode Tensor Batch 和分页 K/V 数据。当前背压会取消并释放请求；后续可以增加自动 Deadline、异步 Writer 暂停/恢复和单线程 ServiceCore 事件循环。真实多请求 Prefill Tensor Batch 仍是 Chapter08 后续能力，不能由 Transport 线程替代。
