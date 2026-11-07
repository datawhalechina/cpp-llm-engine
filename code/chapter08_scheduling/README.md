# Chapter 08 代码：调度与批处理

本章在 Chapter06 Runtime 和 Chapter07 分页 KV Cache 之上选择下一轮要执行的请求。调度器处理的是请求、词元和缓存容量预算，不负责网络接入，也不实现模型算子。先阅读 [`Chapter08 导读`](../../source/chapter08_scheduling/next/intro.md)，再运行：

```bash
bash code/chapter08_scheduling/scripts/run.sh
```

## 资产总览

| 模块 | 学习内容 | 验证 |
| --- | --- | --- |
| `RoundRobinScheduler`、`TensorBatchScheduler` | 从逐请求轮转过渡到教学批执行 | `chapter08_scheduler_test` |
| `StepBatchPlanner`、`ContinuousBatchScheduler` | 在请求数和词元预算内混合 Prefill 与 Decode | `chapter08_step_scheduler_test` |
| `PriorityAgingPolicy` | 优先级、等待老化和稳定的平局顺序 | `chapter08_priority_aging_test` |
| `OverlapScheduler` | 预计算下一批，并在请求状态变化后拒绝旧计划 | `chapter08_overlap_scheduler_test` |
| 缓存事务与批输入 | 预留 KV 槽位、执行、提交或回滚 | `chapter08_cache_transaction_test` |
| Ragged Tensor Batch | 组织不同长度输入并批量执行 | `chapter08_tensor_batch_test` |

## 阅读顺序

1. 从 `scheduler.cpp` 对比 Round-robin 和 Tensor Batch，理解“选择请求”与“合并执行”的区别。
2. 阅读 `StepBatchPlanner::build()`，观察请求数预算、词元预算和 Chunked Prefill 如何共同决定批次。
3. 阅读 `ContinuousBatchScheduler::run_step()`，跟踪缓存预留、模型执行、请求推进和缓存提交。
4. 阅读 `PriorityAgingPolicy`，确认高优先级与长时间等待如何平衡。
5. 最后阅读 `OverlapScheduler`，理解预计算计划为什么必须绑定请求、模型和页表快照。

一次调度步遵循“计划、预留、执行、提交”顺序。计划不修改请求；容量不足时请求留到下一轮；执行失败时释放本轮新增块；只有模型与请求都推进后才提交新的缓存长度。

## 验证

默认脚本运行本章六组测试：

```bash
bash code/chapter08_scheduling/scripts/run.sh
```

有 TinyLlama GGUF 和已构建的 llama.cpp 时，再验证真实批量执行：

```bash
bash code/chapter08_scheduling/scripts/run_real_batch.sh
```

## 扩展位置

当前实现包含同步 CPU 调度、优先级老化和 overlap 计划校验。后续可以增加 Deadline、抢占和设备事件重叠；网络准入由 Chapter10 负责，前缀缓存复用建立在 Chapter07 的块管理之上。
