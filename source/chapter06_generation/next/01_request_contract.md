# 06.1 建立请求与模型执行契约

模型已经能接收词元并产生 logits，但生成过程会跨越多次模型调用。代码需要保存输入、已生成词元、当前位置和结束状态，同时把模型计算与生成控制流分开。Chapter06 用请求（`Request`）、生成配置（`GenerationConfig`）和模型执行器（`RequestModelExecutor`）固定这组交接数据。

## 请求保存哪些状态

请求创建时接收 prompt，随后只通过明确的状态转换提交进度。<a href="/_code/chapter06_generation/include/chapter06_generation/request.h"><code>request.h</code></a> 暴露读取接口和状态转换，<a href="/_code/chapter06_generation/src/request.cpp"><code>request.cpp</code></a> 保证终态不会再次恢复为运行态。

| 数据 | 代码表示 | 变化时机 |
| --- | --- | --- |
| 输入词元 | `prompt()` | 构造请求时固定 |
| Prefill 进度 | `prefill_offset()` | 一段 prompt 成功提交后增加 |
| 已生成词元 | `generated()` | 采样结果成功后追加 |
| 当前阶段 | `phase()` | Prefill、Decode 或结束事件发生时改变 |
| 失败原因 | `failure_reason()` | 模型执行失败时写入 |
| 教学词元历史 | `cache()` | prompt 或生成词元提交后追加 |

`Initial`、`Prefilled` 和 `Decoding` 表示请求仍可推进。`Finished`、`Cancelled`、`TimedOut`、`Rejected` 和 `Failed` 都是终态；终态请求不会再增加 Prefill 游标或生成词元。

## 配置与模型执行器

生成配置定义停止和准入条件，位于 <a href="/_code/chapter06_generation/include/chapter06_generation/generation_types.h"><code>generation_types.h</code></a>。`max_new_tokens` 限制输出长度，`max_context_tokens` 限制 prompt 与计划输出的总长度，`eos_token_id` 提供提前停止条件。

模型执行器只回答“给定输入后 logits 是什么”，不选择下一个词元，也不改变请求终态。接口位于 <a href="/_code/chapter06_generation/include/chapter06_generation/model_executor.h"><code>model_executor.h</code></a>。

| 调用 | 输入 | 输出 | 模型侧状态 |
| --- | --- | --- | --- |
| `prefill(prompt)` | 完整 prompt | 首个待采样 logits | 创建并填充请求上下文 |
| `prefill_chunk(...)` | 一段 prompt | 最后一段才返回 logits | 逐段增加已提交位置 |
| `decode(token)` | 上一步已生成词元 | 下一组 logits | 追加该词元对应的模型 K/V |
| `execution_state()` | 无 | 已提交词元数和 K/V 层数 | 只读观察，不推进模型 |

请求归 Generation Core 管理可观察状态；执行器持有真实模型 Context。教学 `cache()` 当前保存词元 ID，不能代替真实模型的分层 K/V，也不承担物理 Block 分配。

## 验证请求状态

从仓库根目录执行 Request 契约测试：

```bash
cmake -S code/chapter06_generation -B /tmp/cpp-llm-chapter06-contract \
  -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/cpp-llm-chapter06-contract \
  --target chapter06_request_contract_test -j2
ctest --test-dir /tmp/cpp-llm-chapter06-contract \
  -R '^chapter06_request_contract_test$' --output-on-failure
```

通过时可以观察到 prompt 与缓存互不混淆、Prefill 游标单调递增且不越界、五种终态都拒绝后续状态变化。下一篇使用这组契约区分 Prefill 与 Decode。
