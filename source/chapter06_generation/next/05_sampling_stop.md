# 06.5 采样下一个词元并判断停止

模型输出的是每个候选词元的 logits，生成循环需要把它们变成一个词元 ID。采样器（`Sampler`）只负责选择，Runtime 在提交词元后判断 EOS 和长度上限；分开这两个动作可以独立验证数值决策与生命周期。

## Greedy 采样基线

当前默认实现是 Greedy Sampler，代码位于 <a href="/_code/chapter06_generation/src/sampler.cpp"><code>sampler.cpp</code></a>。它返回最大 logit 所在的索引：

```text
logits = [0.1, 0.9, 0.2]
token  = argmax(logits) = 1
```

Greedy 不需要随机数，同一组 logits 始终产生同一个词元，适合作为 Runtime、KV Cache 和批处理重构时的正确性基线。

| 对象 | 决定什么 | 不决定什么 |
| --- | --- | --- |
| 模型执行器 | 返回候选 logits | 不选择词元，不结束请求 |
| `Sampler` | 从 logits 选择词元 ID | 不修改请求状态 |
| `Runtime` | 提交词元并判断停止 | 不改变模型内部 logits 算法 |

Temperature、top-k 和 top-p 尚未进入当前公共实现。增加这些策略时应扩展 Sampler 配置与独立统计测试，不修改 Prefill/Decode 的执行顺序。

## 两个停止条件

Runtime 每次提交采样结果后检查：

1. 词元是否等于 `eos_token_id`；
2. 已生成数量是否达到 `max_new_tokens`。

`max_context_tokens` 在模型执行前做准入检查。如果 prompt 长度加计划输出长度超过上限，请求直接进入 `Rejected`，而不是生成到一半再越界。

| 条件 | 检查时机 | 请求结果 |
| --- | --- | --- |
| EOS | 当前词元提交后 | `Finished`，保留 EOS 词元 |
| 输出长度上限 | 当前词元提交后 | `Finished` |
| 上下文预算不足 | Prefill 前 | `Rejected`，不产生输出 |
| 空 logits 或模型错误 | 采样前 | `Failed`，不产生占位词元 |

## 验证选择和停止

```bash
cmake -S code/chapter06_generation -B /tmp/cpp-llm-chapter06-sampling \
  -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/cpp-llm-chapter06-sampling \
  --target chapter06_sampler_test chapter06_generation_test -j2
ctest --test-dir /tmp/cpp-llm-chapter06-sampling \
  -R 'chapter06_(sampler|generation)_test' --output-on-failure
```

Sampler 测试固定 `[0.1, 0.9, 0.2] -> 1`；Generation 测试把 token `7` 配置为 EOS，确认只生成一个词元就进入 `Finished`。下一篇区分停止判断所见的词元历史与模型真正保存的 K/V。
