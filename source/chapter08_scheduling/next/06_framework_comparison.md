# 08.6 对照成熟框架的调度重点

本教程把请求选择、K/V 资源和模型执行拆成独立对象，便于逐项验证。成熟框架更强调这些对象在设备容量、缓存命中和异步执行下如何协作。

## 对象映射

| 教学对象 | 成熟系统中的方向 | 生产系统继续处理什么 |
| --- | --- | --- |
| `StepBatchPlanner` | iteration/step scheduler | waiting/running 队列、优先级、抢占 |
| token/request budget | token、sequence、KV 容量预算 | 设备容量估计与动态负载 |
| Chunked Prefill | 长输入分段执行 | TTFT/ITL 权衡和动态 chunk |
| Chapter07 缓存接口 | cache manager | 前缀命中、共享、淘汰和分层缓存 |
| Executor batch | model runner input | Attention metadata、异步执行和多设备 |
| `OverlapScheduler` | async/overlap loop | 设备事件、占位输出和流水线状态 |

## 教学实现与生产实现

vLLM V1 使用统一的待计算词元差值覆盖 Prefill、Decode、前缀命中和推测解码；SGLang 把排序与 Radix Cache 命中结合；TensorRT-LLM 先做容量筛选，再组成 MicroBatch；llama.cpp 以 slot 和实用服务器参数组织连续批处理。

| 框架 | 阅读重点 | 本教程对应位置 |
| --- | --- | --- |
| vLLM | 统一 token budget、KV 分配、抢占和异步调度 | 08.2、08.3、08.5 |
| nano-vLLM | 最小 Python Scheduler 与 Paged KV 骨架 | 08.2、08.4 |
| Mini-SGLang | Prefill/Decode 队列和 normal/overlap loop | 08.3、08.5 |
| SGLang | LPM、Radix Cache 和缓存感知排序 | Chapter07、08.3 |
| TensorRT-LLM | Capacity Scheduler 与 MicroBatch Scheduler | 08.2、08.4 |
| llama.cpp | slot、连续批处理和共享上下文容量 | 08.1、08.2 |

## 继续阅读源码

| 项目卡 | 追踪问题 |
| --- | --- |
| [vLLM](/reference/ecosystem/vllm) | Scheduler 输出怎样连接 KV manager 和 model runner |
| [nano-vLLM](/reference/ecosystem/nano_vllm) | 最小代码怎样表达 token budget 和分页位置 |
| [Mini-SGLang](/reference/ecosystem/mini_sglang) | normal loop 与 overlap loop 的事件顺序 |
| [SGLang](/reference/ecosystem/sglang) | LPM 怎样利用前缀命中并受缓存预算约束 |
