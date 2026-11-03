# 06.1 把词元变成带位置信息的模型输入

生成 runtime 接收 token id，但 Decoder 计算需要连续数值表示。输入路径先完成 embedding，再注入位置信息。Llama 使用 RoPE 作用于 Attention 的 Q/K；Chapter06 的小型 smoke model 使用更简单的确定性位置增量，以便先验证 runtime 调用顺序。

先从 <a href="../_code/chapter06_generation/src/embedding.cpp"><code>embedding.cpp</code></a> 观察 smoke model 的调用顺序，再与真实 TinyLlama 的 embedding 和 RoPE 路径比较。

## 输入表示与两条执行路径

标准 embedding table 的逻辑形状是 `[vocab_size, hidden_size]`。输入 `[B, S]` 中的每个 token id 选择一行，得到 `[B, S, D]` hidden state。特殊 token 与普通 token 使用相同查表规则；EOS 的特殊含义由 runtime 停止条件解释。

位置方案改变的是“顺序如何进入计算”，不能只凭类名推断模型采用哪一种实现。

| 方案 | 位置进入计算的位置 | 常见模型 |
| --- | --- | --- |
| Sinusoidal | 与 embedding 相加 | 原始 Transformer |
| Learned position | 查询位置参数并与 embedding 组合 | BERT、GPT-2 |
| RoPE | 在 Attention 中旋转 Q/K | Llama、TinyLlama |

Chapter06 的 `Embedding`、`PositionEncoding` 和 `Transformer` 是 runtime smoke model。它们用 `Logits`，即 `std::vector<float>`，保存每个 token 对应的确定性标量；`Embedding::forward()` 做 token 归一化，`PositionEncoding::apply()` 加上位置值。它们验证 prefill/decode、状态推进和停止条件，不验证真实 embedding 矩阵或 RoPE 数值。

真实 TinyLlama 路径由 `RealLlamaCpuModel` 读取 `token_embd.weight`，并由 `RealLlamaCpuContext` 在每层 Attention 中应用 RoPE。该 Context 被 Chapter06 的 `RequestModelExecutor` 适配器调用，因此 runtime contract 不需要知道底层是 smoke model 还是真实模型。

| 路径 | 数据表示 | 位置信息 | 验证目标 |
| --- | --- | --- | --- |
| Chapter06 smoke model | 每 token 一个确定性 `float` | `position_base + i` 的数值增量 | runtime 编排与状态机 |
| TinyLlama CPU Executor | hidden size 2048 的向量 | 每层 Q/K 上的 RoPE | 真实 GGUF 模型前向 |

## Token 到模型输入的调用链

仓库中的可执行调用顺序如下。`Transformer::forward()` 会把 prompt 与已生成 token 合并，再调用 embedding 和 position；代码片段只保留关键语句。

```cpp
TokenSequence tokens = request.prompt();
tokens.insert(tokens.end(), request.generated().begin(), request.generated().end());

Logits hidden = embedding_.forward(tokens);
position_encoding_.apply(hidden, step);
```

`prefill()` 使用位置基准 0；`decode()` 使用 `request.generated_size()` 作为 smoke 路径的位置基准。真实 Executor 则在请求独占的 `RealLlamaCpuContext` 中维护准确的 token position 和分层 K/V。

## 验证与衔接

Chapter06 测试固定 `Embedding(32)`、`PositionEncoding(16)` 和两层 smoke Transformer，检查组件 ready、层数、prefill token 数、decode 次数和终态。真实路径的 RoPE、分层 K/V 与 golden token 由 Engine smoke 覆盖。

| 检查项 | 对应证据 |
| --- | --- |
| smoke embedding/position 已接入 | `chapter06_generation_test.cpp` |
| prompt 与 generated token 按顺序进入 forward | `transformer.cpp` |
| TinyLlama 使用真实 embedding 与 RoPE | `real_llama_cpu_decoder.cpp` |
| prefill/decode 位置和 K/V 连续 | `real_tinyllama_generation_smoke` |

验证通过后进入 [`06.2 N 层 Transformer 组装`](./02_transformer_assembly.md)，观察同一 hidden state 怎样连续经过多层 Decoder。
