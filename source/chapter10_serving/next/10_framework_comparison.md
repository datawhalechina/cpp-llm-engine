# 10.10 对照成熟推理服务架构

教学 ServiceCore 把一个进程内的请求生命周期拆清楚。成熟框架保留这些对象，同时增加多进程 Worker、完整 API、分布式缓存、租户策略和故障恢复。

## 对象映射

| 教学对象 | llama.cpp server | vLLM | SGLang |
| --- | --- | --- | --- |
| Transport/Adapter | endpoint 与 server context | OpenAI API server | HTTP server/tokenizer manager |
| admission/waiting | slot 分配与状态 | EngineCore request | Scheduler request |
| Chapter08 Scheduler | slot 推进 | V1 Scheduler | SRT Scheduler |
| Executor/Context | model/context | worker/model runner | model worker |
| K/V 所有权 | context KV | KV cache manager | radix/prefix cache |
| metrics | timings/server metrics | serving metrics | scheduler/server metrics |

## 教学实现与生产实现

| 问题 | 成熟系统增加的对象 | 工程影响 |
| --- | --- | --- |
| API 与 Engine 分进程 | IPC、序列化、Engine client | 取消和流式事件跨进程传递 |
| 多 GPU/多节点执行 | distributed executor、worker group | 资源与失败域扩大 |
| 多租户和优先级 | policy、quota、preemption | admission 与 Scheduler 策略分离 |
| 前缀与分层缓存 | prefix index、host/remote cache | 生命周期不再等同单请求 |
| 生产观测 | histogram、trace、request log | 指标基数与隐私治理 |

## 继续阅读源码

阅读时先跟踪一条请求如何进入 Engine、获得 K/V、产生首词元并在取消后释放资源，再观察 API 进程和模型 Worker 分开后这些事件如何跨进程传递。

| 项目卡 | 本章使用它回答什么 |
| --- | --- |
| [llama.cpp](../../reference/ecosystem/llama_cpp.md) | 单机 server 如何组织 slot、Context 和流式输出 |
| [vLLM](../../reference/ecosystem/vllm.md) | API server、EngineCore 和 model runner 怎样分工 |
| [SGLang](../../reference/ecosystem/sglang.md) | tokenizer manager、Scheduler 和 radix cache 怎样交接 |
| [TensorRT-LLM](../../reference/ecosystem/tensorrt_llm.md) | Executor、Worker 和多 GPU 执行怎样扩展故障域 |
