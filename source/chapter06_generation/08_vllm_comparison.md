# 06.8 对照成熟框架的生成运行时

Chapter06 把 `prefill -> decode -> sample -> output` 做成可观察状态机。成熟 Runtime 在相同链路上增加动态批处理、分页缓存、前缀复用、抢占和异步执行。

## 对象映射

Continuous Batching、请求排序、KV 物理管理和前缀复用会在一次 schedule step 中协作，但分别回答不同问题。

| 教程对象 | 成熟框架中的对应物 | 阅读时追踪的问题 |
| --- | --- | --- |
| `StepBatchPlanner` | vLLM iteration scheduler、SGLang Prefill/Decode manager | 一个 step 输出请求列表、token 数还是完整执行 metadata |
| admission 顺序与最少进度优先 | priority、FCFS、LPM 与抢占策略 | waiting 请求怎样进入运行集合并避免饥饿 |
| `KvBlockPool` / `BlockTable` | KV cache manager、block/page pool | Scheduler 与 cache manager 分别持有什么状态 |
| 请求级 Model Context | prefix caching、Radix Cache / RadixAttention | 已计算前缀怎样匹配、引用和释放 |
| 同步 `run_step()` | async engine、overlap scheduling | CPU 调度与设备执行怎样重叠 |

## 教学实现与生产实现

教学 Runtime 用确定性 CPU step 验证状态迁移和资源回收；生产系统继续处理设备页、跨请求复用和异步并发。

| 本章可验证对象 | 成熟系统继续增加 | 增加它解决什么问题 |
| --- | --- | --- |
| 单请求和多请求 prefill/decode | 抢占、优先级和 speculative decode | 提升负载适应性和解码吞吐 |
| 最少进度优先与 token budget | priority、LPM 和 overlap scheduling | 控制公平性、TTFT 和设备空闲时间 |
| CPU block pool 与 block table | device K/V payload、PagedAttention | 降低长上下文内存浪费和搬迁 |
| callback streaming | HTTP/SSE、背压和断连 | 支持真实客户端和慢消费者 |

## 继续阅读源码

从项目卡进入源码后，跟踪一次 `选择请求 -> 分配 cache -> 执行模型 -> 更新状态`：

1. Scheduler 输出请求列表、token 数，还是完整执行 metadata？
2. Cache manager 和 Scheduler 分别持有哪些 block 状态，失败时谁负责回收？
3. Chunked Prefill 加入后，公平性和 TTFT 为什么可能同时变化？
4. LPM 改变 waiting 顺序时，怎样使用 Radix Cache 的 prefix match 结果？

| 项目卡 | 在本章中的用途 |
| --- | --- |
| [vLLM](../reference/ecosystem/vllm.md) | Continuous Batching、Block KV 和 EngineCore |
| [Mini-SGLang](../reference/ecosystem/mini_sglang.md) | 紧凑的 Prefill/Decode 与 overlap 调度骨架 |
| [SGLang](../reference/ecosystem/sglang.md) | LPM、Radix Cache 和多级缓存扩展 |
