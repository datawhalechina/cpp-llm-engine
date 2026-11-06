# FQ09 Request State

## 任务

实现一个最小请求状态机，表达 generation runtime 中 `waiting -> prefill -> running -> finished/aborted` 的合法迁移。重点是请求何时允许进入下一阶段，以及终态如何拒绝后续操作。

## 完成标准

- 合法迁移能按顺序执行。
- 非法迁移明确失败，不静默修改状态。
- 达到 `max_new_tokens` 后自动进入 `finished`。
- 取消请求进入 `aborted`，终态请求不能再次运行。

## 验证

运行 `bash exercises/foundation/run_all.sh`，重点测试 `fq09_request_state`；再对照 Chapter 06 的 Request 生命周期和 scheduler admission 规则。

## 提示

- L0：先列出状态和允许的边。
- L1：让每个动作先检查当前状态，再修改状态。
- L2：为取消、重复完成和超出生成上限增加测试。

## 主线映射

对应 [`Chapter06 Request 生命周期`](/chapter06_generation/next/04_request_lifecycle) 和 [`Chapter08 调度基线`](/chapter08_scheduling/next/01_scheduler_baseline)。
