# Chapter 08 代码资产：调度与批处理

本目录从 Chapter06 Runtime 和 Chapter07 缓存接口之上组织多请求执行。学习说明见 [`Chapter08 导读`](../../source/chapter08_scheduling/next/intro.md)。先运行：

```bash
bash code/chapter08_scheduling/scripts/run.sh
```

有 TinyLlama GGUF 和已构建的 llama.cpp 时，只验证真实批量执行：

```bash
bash code/chapter08_scheduling/scripts/run_real_batch.sh
```

## 资产总览

| 资产 | 职责 | 验证 |
| --- | --- | --- |
| `scheduler.h/.cpp` | Round-robin 与教学 Tensor Batch 基线 | `chapter08_scheduler_test` |
| `step_scheduler.h/.cpp` | StepBatchPlan、Continuous Batching、Executor batch 输入 | `chapter08_step_scheduler_test` |
| `priority_aging_test.cpp` | 基础优先级、有界老化与确定性平局规则 | `chapter08_priority_aging_test` |
| `overlap_scheduler.h/.cpp` | 预计算计划、状态快照、失效与同步回退 | `chapter08_overlap_scheduler_test` |
| `cache_transaction_test.cpp` | 调度、模型位置和 Chapter07 BlockTable 的事务 | `chapter08_cache_transaction_test` |
| `tensor_batch_test.cpp` | Ragged 输入、批量 Prefill 和逐请求准入 | `chapter08_tensor_batch_test` |

当前实现包含同步 CPU 调度、优先级老化和 overlap 计划控制面。网络 admission 属于 Chapter10；Prefix Cache 属于 Chapter07；Deadline、抢占和设备事件重叠尚未实现。
