# FQ16 Prefill/Decode State

## 任务

实现生成执行的最小阶段状态机：`Waiting -> Prefill -> Decode -> Finished`，并支持在非终态取消。重点是区分一次 prompt prefill 和逐 token decode；Prefill 只完成阶段切换，Decode 每次推进一个生成 token。

## 完成标准

- 新请求从 `Waiting` 开始。
- admission 只能进入 `Prefill`。
- prefill 完成后才能进入 `Decode`。
- decode 每次只增加一个 generated token。
- 达到 `max_new_tokens` 后进入 `Finished`。
- 非终态请求可以进入 `Aborted`，终态请求不能重复迁移。
- 非法状态迁移抛出明确异常。

## 验证

复制并完成 starter 后，先运行自己的答案：

```bash
cp -r exercises/foundation/problems/fq16_prefill_decode_state/starter /tmp/fq16-answer
bash scripts/check_foundation_answer.sh /tmp/fq16-answer/main.cpp
```

运行 `bash exercises/foundation/run_all.sh`，重点检查 `fq16_prefill_decode_state` 的正常、完成和取消路径。

## 提示

- L0：先列出阶段和允许的迁移表。
- L1：把 `generated_tokens` 和 `max_new_tokens` 放在状态对象中。
- L2：所有公开迁移函数先检查当前阶段，再修改阶段或计数。

## 主线映射

对应 [`Chapter06 Request 生命周期`](/chapter06_generation/next/04_request_lifecycle)、[`单请求生成循环`](/chapter06_generation/next/03_single_request_loop) 和 [`Chapter10 终态释放`](/chapter10_serving/next/03_terminal_release)。
