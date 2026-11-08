# FQ13 Shape-aware Batch

## 任务

根据一组 request 的 sequence length，计算一次 batch 的最大长度、有效 token 数、padding 后的 token 数，以及每一行在连续 batch buffer 中的起始 offset。

## 完成标准

- 拒绝长度为零的 request。
- 按容量限制接收 request。
- 能区分 `active_tokens` 和 `padded_tokens`。
- 能根据 batch 行号计算连续 buffer offset。
- 越界行号明确失败。

## 验证

复制并完成 starter 后，先运行自己的答案：

```bash
cp -r exercises/foundation/problems/fq13_shape_aware_batch/starter /tmp/fq13-answer
bash scripts/check_foundation_answer.sh /tmp/fq13-answer/main.cpp
```

运行 `bash exercises/foundation/run_all.sh`，重点检查 `fq13_shape_aware_batch`。

## 提示

- L0：先保存每个 request 的 sequence length。
- L1：`max_sequence_length` 决定每一行的 stride。
- L2：`active_tokens` 用于真实工作量，`padded_tokens` 用于 dense batch buffer 的容量估算。

## 主线映射

对应 [`Chapter06 Generation Core`](/chapter06_generation/next/intro)、[`Request 生命周期`](/chapter06_generation/next/04_request_lifecycle) 和 [`Chapter08 Step Batching`](/chapter08_scheduling/next/02_step_batching)。
