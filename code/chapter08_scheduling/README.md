# Chapter 08 代码：调度与批处理

本章在 Chapter06 Runtime 和 Chapter07 分页 KV Cache 之上选择下一轮要执行的请求。调度器处理的是请求、词元和缓存容量预算，不负责网络接入，也不实现模型算子。先阅读 [`Chapter08 导读`](../../source/chapter08_scheduling/next/intro.md)，再运行：

```bash
bash code/chapter08_scheduling/scripts/run.sh
```

## 代码架构

调用方拥有 `Request`、`Sampler` 和请求专属 Executor；Chapter08 只保存策略状态或本步临时计划。同步执行按以下顺序连接 Chapter06 与 Chapter07：

```text
ScheduledRequest[]
       │
       ▼
StepBatchPlanner ── PriorityAgingPolicy
       │           └── cached_prefix_tokens
       │ StepBatchPlan（只读候选）
       ▼
ContinuousBatchScheduler
       ├── KvCacheTransaction：reserve / commit / trim
       ├── ExecutorBatchInput：token / position / request ID / K/V slot
       ├── ExecutorBatchRunner：真实 Decode batch
       └── Runtime：提交 logits、采样并更新 Request
```

Planner 不修改请求和缓存；同优先级、同等待时间的 Prefill 请求优先选择 `cached_prefix_tokens` 更长者。该字段只提供排序证据，不会自动推进 Request 或模型 Context。Scheduler 在执行前预留资源，模型失败时裁剪到原计数，成功后才提交 K/V 与请求结果。Runner 必须自行回滚模型 Context，因为 Scheduler 只拥有请求与外部缓存事务。`OverlapScheduler` 只保存计划和状态快照，不拥有 Request、Executor 或物理块。

## 阅读顺序与资产

| 模块 | 学习内容 | 验证 |
| --- | --- | --- |
| `RoundRobinScheduler`、`TensorBatchScheduler` | 从逐请求轮转过渡到教学批执行 | `chapter08_scheduler_test` |
| `StepBatchPlanner`、`ContinuousBatchScheduler` | 在请求数和词元预算内混合 Prefill 与 Decode | `chapter08_step_scheduler_test` |
| `PriorityAgingPolicy`、`cached_prefix_tokens` | 优先级、等待老化、缓存命中和平局顺序 | `chapter08_priority_aging_test`、`chapter08_step_scheduler_test` |
| `OverlapScheduler` | 预计算下一批，并在请求状态变化后拒绝旧计划 | `chapter08_overlap_scheduler_test` |
| 缓存事务与批输入 | 预留 KV 槽位、执行、提交或回滚 | `chapter08_cache_transaction_test` |
| Ragged Tensor Batch | Chapter06 教学 Transformer 的批量 Prefill 前置能力 | `chapter08_tensor_batch_test`（integration） |
| `ExecutorBatchInput`、`ExecutorBatchRunner` | 把 Decode 候选交给真实模型批量执行 | `chapter08_cache_transaction_test`、真实模型测试 |

1. 从 `scheduler.cpp` 对比 Round-robin 和 Tensor Batch，理解“选择请求”与“合并执行”的区别。
2. 阅读 `StepBatchPlanner::build()`，观察请求数预算、词元预算、缓存命中和 Chunked Prefill 如何共同决定批次。
3. 阅读 `ExecutorBatchInput` 和 `ContinuousBatchScheduler::run_step()`，跟踪缓存预留、批输入、模型执行、请求推进和缓存提交。
4. 阅读 `PriorityAgingPolicy`，确认高优先级与长时间等待如何平衡。
5. 最后阅读 `OverlapScheduler`，理解预计算计划为什么必须绑定请求、模型和页表快照。

一次调度步遵循“计划、预留、执行、提交”顺序。计划不修改请求；容量不足时请求留到下一轮；执行失败时缓存裁剪到执行前词元数；只有模型与请求都推进后才提交新的缓存长度。

## 运行与验证

默认脚本运行本章六组测试：

```bash
bash code/chapter08_scheduling/scripts/run.sh
```

有 TinyLlama GGUF 和已构建的 llama.cpp 时，再验证真实 Decode batch：

```bash
bash code/chapter08_scheduling/scripts/run_real_batch.sh
```

## 扩展入口

当前实现包含同步 CPU 调度、优先级老化、缓存命中排序、真实 Decode batch 和 overlap 计划校验。Chapter07 负责产生最长前缀命中并挂接共享块；模型 Context 采用命中、真实多请求 Prefill Tensor batch、Deadline、抢占和设备事件重叠仍待扩展。网络准入由 Chapter10 负责。
