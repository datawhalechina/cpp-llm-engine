# Chapter 10 代码：推理服务

本章把 Chapter06 请求状态、Chapter07 KV 资源和 Chapter08 调度器接入同一个服务生命周期，并通过与网络库无关的接口返回同步或流式结果。先阅读 [`Chapter10 导读`](../../source/chapter10_serving/next/intro.md)，再运行：

```bash
bash code/chapter10_serving/scripts/run.sh
```

## 资产总览

| 模块 | 学习内容 | 验证 |
| --- | --- | --- |
| `ServiceCore` | 准入、等待/运行队列、调度交接、指标和统一释放 | `chapter10_service_core_test` |
| `RequestWorkerPool` | 用独立 Request 与 Executor 并行执行 CPU 请求 | `chapter10_request_worker_test` |
| `ResponseWriter` | 表达响应头、SSE 事件、断连和背压 | `chapter10_http_contract_test` |
| `HttpAdapter` | 把 health、metrics 和 Completion 路由映射到 ServiceCore | `chapter10_http_contract_test` |
| `CppHttplibServer` | 把可选 cpp-httplib socket 接到 HttpAdapter | `run_cpp_httplib_loopback.sh` |
| 终态释放 | 从 waiting/running 移除请求并释放 K/V 与批执行状态 | `chapter10_terminal_release_test` |

## 阅读顺序

1. 阅读 `ServiceCore::admit()` 和 `promote_waiting()`，理解服务容量与调度预算的区别。
2. 阅读 `dispatch_step()`，观察服务如何把运行请求交给 Chapter08，而不是重新实现调度。
3. 阅读 `release_terminal()`，确认完成、取消、超时和失败共用一条资源释放路径。
4. 阅读 `HttpAdapter` 与 `ResponseWriter`，跟踪同步 Completion 和 SSE 如何处理断连、背压与异常。
5. 最后阅读 `CppHttplibServer`，查看网络库如何只承担 Transport 适配。

请求从 waiting 进入 running 后，由 Chapter08 每步选择执行对象。请求到达终态并不等于资源已经释放；调用 `release_terminal()` 后，Request 缓存、批执行绑定和分页 KV 块才会一起回收。

## 验证

默认脚本不访问网络，运行服务状态、HTTP 契约、并行 Worker 和终态释放测试：

```bash
bash code/chapter10_serving/scripts/run.sh
```

## 可选网络闭环

准备本地 `httplib.h` 后，可设置 `CHAPTER10_CPPHTTPLIB_INCLUDE_DIR` 运行 `scripts/run_cpp_httplib_loopback.sh`；未设置时，CMake 获取固定版本 `v0.18.3`。

## 扩展位置

ServiceCore 已接通 Chapter08 优先级策略、真实 Tensor Batch 和分页 KV 数据。当前背压会取消并释放请求；后续可以增加自动 Deadline、异步 Writer 暂停/恢复和更细粒度并发 Transport，而不改变 ServiceCore 的请求生命周期。
