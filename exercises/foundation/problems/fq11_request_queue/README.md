# FQ11 Request Queue

## 任务

实现一个带容量上限的请求队列，模拟 scheduler admission、FIFO 推进和取消后的 backpressure 行为。

## 完成标准

- 队列不超过容量上限。
- 容量不足时 admission 返回明确结果，不丢失已有请求。
- 未取消请求按 FIFO 顺序出队。
- 取消请求不会再次被 pop。
- 空队列 pop 有明确结果。

## 验证

复制并完成 starter 后，先运行 `bash scripts/check_foundation_answer.sh /tmp/fq11-answer/main.cpp`。

运行 `bash exercises/foundation/run_all.sh`，重点测试 `fq11_request_queue`；再对照 Chapter 06 的 scheduler admission 和 continuous batching 规则。

## 提示

- L0：先固定容量不足和空队列的返回契约。
- L1：使用 `std::deque` 保存等待请求，取消时只移除匹配项。
- L2：增加重复 ID、取消队首、取消队尾和容量释放后的重新 admission 测试。

## 主线映射

对应 [`Chapter08 调度基线`](/chapter08_scheduling/next/01_scheduler_baseline)、[`Step Batching`](/chapter08_scheduling/next/02_step_batching) 和 [`推理引擎服务化基础`](/reference/service_basics)。
