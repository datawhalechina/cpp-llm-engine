# FQ07 Token Queue

## 任务

实现一个 FIFO token queue，模拟 generation runtime 中等待处理的 token 或请求事件。

## 完成标准

- `push` 按顺序追加 token。
- `pop` 按 FIFO 顺序返回 token。
- 空队列 `pop` 有明确失败行为。
- `size` 始终反映队列状态。

## 验证

复制并完成 starter 后，先运行 `bash scripts/check_foundation_answer.sh /tmp/fq07-answer/main.cpp`。

运行 `bash exercises/foundation/run_all.sh`，重点测试 `fq07_token_queue`。

## 提示

- L0：先确定空队列的契约。
- L1：使用 `std::deque` 或两个栈实现 FIFO。
- L2：增加交错 push/pop 和空队列恢复测试。

## 主线映射

对应 [`Foundation 数据结构与算法基础`](/foundation/06_data_structure_algorithm_basics) 和 [`Chapter08 调度基线`](/chapter08_scheduling/next/01_scheduler_baseline)。
