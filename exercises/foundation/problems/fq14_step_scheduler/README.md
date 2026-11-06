# FQ14 Step Scheduler

## 任务

实现一个确定性的 decode scheduler：每次 step 最多选择 `max_batch` 个 request，并且最多消耗 `token_budget` 个 token。每个 request 执行一个 token 后，未完成 request 回到队尾。

## 完成标准

- 按 FIFO 顺序选择 request。
- 同一个 step 同时受 batch 上限和 token budget 限制。
- 未完成 request 重新进入队尾，形成 round-robin 推进。
- 已完成 request 不再进入下一轮。
- `max_batch` 或 `token_budget` 为零时不执行任何 request。

## 验证

运行 `bash exercises/foundation/run_all.sh`，重点检查 `fq14_step_scheduler` 的三轮执行顺序。

## 提示

- L0：队列元素需要同时保存 request id 和剩余 token 数。
- L1：每次选择一个 request，只执行一个 decode token。
- L2：先消耗 token budget，再检查 batch 上限；两者都满足时才能继续选择。

## 主线映射

对应 [`Chapter08 分块与优先级`](/chapter08_scheduling/next/03_chunked_priority)、[`Chapter06 Request 生命周期`](/chapter06_generation/next/04_request_lifecycle) 和 [`Step Batching`](/chapter08_scheduling/next/02_step_batching)。
