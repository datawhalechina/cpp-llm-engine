# 06.9 对照成熟框架的生成核心

## 对象映射

Chapter06 只比较单请求生成核心：输入怎样进入模型、Prefill/Decode 怎样推进、采样结果怎样成为输出。KV 分页、跨请求调度和网络服务分别在 Chapter07、Chapter08 和 Chapter10 对照。

| 教程对象 | llama.cpp | vLLM | TensorRT-LLM | 阅读问题 |
| --- | --- | --- | --- | --- |
| `Request` | sequence/slot 的请求状态 | request/sequence data | executor request | 输入、输出和终态保存在哪里 |
| `RequestModelExecutor` | model context 与 decode API | model runner | runtime execution context | 单请求模型状态由谁持有 |
| `Runtime::generate` | decode loop | EngineCore 执行链的一部分 | Executor 执行链的一部分 | Prefill、采样和 Decode 怎样衔接 |
| `Sampler` | sampling 参数与实现 | sampler/model output | decoding 配置 | logits 决策是否与模型执行分离 |
| `TokenCallback` | token output callback | engine output | response/iteration result | 词元在什么时刻对上层可见 |

## 教学实现与生产实现

教程把单请求状态和调用顺序写成直接的 C++ 对象，便于逐步检查。成熟系统还要处理异步执行、设备队列、多个采样参数、结构化输出和跨进程结果传递。

| 工程问题 | Chapter06 的处理 | 成熟系统增加的机制 |
| --- | --- | --- |
| 模型状态 | 每个执行器持有一份请求 Context | Worker、设备执行上下文和异步生命周期 |
| 采样 | 确定性 Greedy | temperature、top-k、top-p、约束解码 |
| 输出 | 同步 token callback | 异步队列、序列化和客户端背压 |
| 错误 | Request 进入稳定终态 | Worker 故障、重试、跨进程错误映射 |
| 资源 | 只描述单请求语义 | KV allocator、抢占和容量预算 |

生产系统的资源和吞吐机制不能倒灌进 Generation Core：Paged KV 改变存储方式，Scheduler 改变请求选择，Serving 改变接入方式，但同一请求的词元顺序和停止语义必须保持一致。

## 继续阅读源码

- [llama.cpp 生态卡](/reference/ecosystem/llama_cpp)：从模型 Context 和本地 decode loop 观察最小运行时。
- [vLLM 生态卡](/reference/ecosystem/vllm)：先看 EngineCore 与 model runner，KV manager 和 Scheduler 留到后续章节。
- [TensorRT-LLM 生态卡](/reference/ecosystem/tensorrt_llm)：观察 C++ Runtime、执行上下文和 Executor 的关系。
- [SGLang 生态卡](/reference/ecosystem/sglang)：本章只看 Runtime 请求推进，Radix Cache 与 LPM 分别放到 Chapter07/08。

## 验证与衔接

```bash
bash code/chapter06_generation/scripts/run_generation_core.sh
```

Request 契约测试验证终态，Generation component 验证 Prefill、Decode、停止和回调顺序；这些结果对应成熟框架在调度与缓存之外仍需保持的单请求语义。[`Chapter07`](../../chapter07_kv_cache/next/intro.md) 将在不改变这些语义的前提下，把模型 K/V 放入分页物理块。
