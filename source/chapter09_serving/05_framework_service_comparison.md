# 09.5 对照成熟 Serving 框架

本章用 C++ `ServiceCore` 连接请求接入、调度、执行、流式输出和资源释放。成熟服务把这些对象进一步拆到 API server、engine core、worker 和 cache manager。

## 对象映射

| 教程对象 | vLLM | SGLang | llama.cpp server |
| --- | --- | --- | --- |
| HTTP/SSE transport | OpenAI-compatible API server | HTTP server / tokenizer manager | endpoint 与 slot |
| `ServiceCore::admit` | engine request admission | scheduler request admission | request/slot 分配 |
| waiting/running 队列 | V1 scheduler | SRT scheduler | slot 状态推进 |
| `ModelExecutorFactory` | executor、worker、model runner | model worker | server context、model context |
| block ownership 与回收 | KV cache manager | radix/prefix cache | context KV cache |
| metrics 与 benchmark JSON | serving metrics | scheduler/server metrics | timings 与 server metrics |

## 教学实现与生产实现

本章验证单实例服务生命周期和指标来源；成熟系统继续增加多进程执行、租户策略、完整 API 和分布式观测。

| 本章可验证对象 | 成熟系统继续增加 | 工程影响 |
| --- | --- | --- |
| admission、waiting/running、cancel、timeout | 多租户优先级、抢占和全局路由 | 调度状态与业务策略分离 |
| 同步完成和 SSE token event | 完整 OpenAI schema、鉴权、TLS | transport 与 Engine contract 分离 |
| 每请求 Executor/Context 与统一回收 | 多进程 worker、分布式 executor | 跨进程资源所有权和失败恢复 |
| CPU worker 请求级并发 | 设备 worker、异步执行与批量模型调用 | 区分请求线程并发和 Tensor batch |
| queue wait、TTFT、completion 聚合 | histogram、P99、分布式 trace | 指标聚合与 trace context |
| token/block-budget dispatch | GPU 并行 batch executor | 计划 metadata 与设备执行连接 |

## 继续阅读源码

从项目卡进入源码后，沿一条请求从 API 到 Engine，再跟踪到终态释放：

1. transport 断开、取消或超时时，哪个对象写入终态并触发回收？
2. Scheduler 选择请求时，KV block 和模型执行资源由谁持有？
3. API server、EngineCore 和 worker 分进程后，哪些 contract 跨进程传递？
4. 前缀命中、waiting 排序和 KV 存储层级分别由哪个对象决定？

| 项目卡 | 在本章中的用途 |
| --- | --- |
| [llama.cpp](../reference/ecosystem/llama_cpp.md) | 本地 C++ 模型上下文、slot 和 HTTP 服务 |
| [vLLM](../reference/ecosystem/vllm.md) | API server、EngineCore、Scheduler 和 KV manager |
| [Mini-SGLang](../reference/ecosystem/mini_sglang.md) | 可读的 normal/overlap 调度循环 |
| [SGLang](../reference/ecosystem/sglang.md) | Scheduler、Radix Cache 和生产服务扩展 |
