# chapter08_scheduling 代码导航

本章代码在 Chapter06 Runtime 和 Chapter07 KV 资源之上组织多请求执行。

## 阅读顺序

1. 先看 `scheduler.h/.cpp`，建立 Round-robin 与 Tensor Batch 基线
2. 再看 `step_scheduler.h/.cpp`，跟踪预算、候选、`ExecutorBatchInput`、优先级老化与 Continuous Batching
3. 最后看六组测试，核对结果等价、公平性、缓存事务和计划失效

## 调度与批处理

| 文件 | 作用 |
| --- | --- |
| <a href="https://github.com/datawhalechina/cpp-llm-engine/blob/main/code/chapter08_scheduling/include/chapter08_scheduling/scheduler.h"><code>scheduler.h</code></a> | `ScheduledRequest`、Round-robin 和 Tensor Batch 接口 |
| <a href="https://github.com/datawhalechina/cpp-llm-engine/blob/main/code/chapter08_scheduling/src/scheduler.cpp"><code>scheduler.cpp</code></a> | 固定轮转与教学批量执行 |
| <a href="https://github.com/datawhalechina/cpp-llm-engine/blob/main/code/chapter08_scheduling/include/chapter08_scheduling/step_scheduler.h"><code>step_scheduler.h</code></a> | `StepBatchPlan`、`ExecutorBatchInput`、优先级老化和预算契约 |
| <a href="https://github.com/datawhalechina/cpp-llm-engine/blob/main/code/chapter08_scheduling/src/step_scheduler.cpp"><code>step_scheduler.cpp</code></a> | Continuous Batching、Decode batch 交接、Chunked Prefill 和缓存事务 |
| <a href="https://github.com/datawhalechina/cpp-llm-engine/blob/main/code/chapter08_scheduling/include/chapter08_scheduling/overlap_scheduler.h"><code>overlap_scheduler.h</code></a> | 预计算批次、状态快照和失效结果 |
| <a href="https://github.com/datawhalechina/cpp-llm-engine/blob/main/code/chapter08_scheduling/src/overlap_scheduler.cpp"><code>overlap_scheduler.cpp</code></a> | 状态校验与同步回退 |

## 验证与脚本

| 文件 | 作用 |
| --- | --- |
| <a href="https://github.com/datawhalechina/cpp-llm-engine/blob/main/code/chapter08_scheduling/tests/scheduler_test.cpp"><code>scheduler_test.cpp</code></a> | Round-robin、Tensor Batch 和请求隔离 |
| <a href="https://github.com/datawhalechina/cpp-llm-engine/blob/main/code/chapter08_scheduling/tests/step_scheduler_test.cpp"><code>step_scheduler_test.cpp</code></a> | token budget、混排与 Chunked Prefill |
| <a href="https://github.com/datawhalechina/cpp-llm-engine/blob/main/code/chapter08_scheduling/tests/priority_aging_test.cpp"><code>priority_aging_test.cpp</code></a> | 基础优先级、有界老化和稳定平局规则 |
| <a href="https://github.com/datawhalechina/cpp-llm-engine/blob/main/code/chapter08_scheduling/tests/cache_transaction_test.cpp"><code>cache_transaction_test.cpp</code></a> | Chapter06/07/08 跨章事务 |
| <a href="https://github.com/datawhalechina/cpp-llm-engine/blob/main/code/chapter08_scheduling/tests/tensor_batch_test.cpp"><code>tensor_batch_test.cpp</code></a> | Chapter06 Ragged Prefill Batch 的跨章前置证据 |
| <a href="https://github.com/datawhalechina/cpp-llm-engine/blob/main/code/chapter08_scheduling/tests/overlap_scheduler_test.cpp"><code>overlap_scheduler_test.cpp</code></a> | 同步等价、计划失效和关闭回退 |
| <a href="https://github.com/datawhalechina/cpp-llm-engine/blob/main/code/chapter08_scheduling/scripts/run.sh"><code>scripts/run.sh</code></a> | Chapter08 独立构建与 CTest 入口 |
| <a href="https://github.com/datawhalechina/cpp-llm-engine/blob/main/code/chapter08_scheduling/scripts/run_real_batch.sh"><code>scripts/run_real_batch.sh</code></a> | TinyLlama 真实 Decode Batch 验证入口 |
