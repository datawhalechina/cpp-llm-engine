# 06.2 区分 Prefill 与 Decode

同一个请求会以两种形状进入模型。Prefill 一次接收多个 prompt 词元并建立上下文；Decode 每次接收一个已生成词元，在已有上下文后追加计算。二者都产生 logits，但输入规模、模型状态和调用次数不同。

## 两种模型调用

下面用 prompt `[1, 2, 3]` 和两个生成词元说明调用顺序。首个输出词元直接从 Prefill 返回的 logits 采样，因此第一次 Decode 消费的是这个刚生成的词元。

| 顺序 | 模型调用 | 本次输入 | 调用后的已提交模型位置 | 产生的结果 |
| --- | --- | --- | --- | --- |
| 1 | Prefill | `[1, 2, 3]` | 3 | 第一个待采样 logits |
| 2 | Sample | Prefill logits | 3 | 第一个生成词元 `g0` |
| 3 | Decode | `g0` | 4 | 第二个待采样 logits |
| 4 | Sample | Decode logits | 4 | 第二个生成词元 `g1` |

`Request::generated()` 记录已经交给调用者的输出，模型执行器的 `committed_tokens` 记录已经进入模型 Context 的输入位置。采样出 `g1` 后，如果请求立即结束，`g1` 不需要再次 Decode；这两个计数不必始终相等。

## Runtime 怎样提交结果

<a href="/_code/chapter06_generation/src/runtime.cpp"><code>runtime.cpp</code></a> 中的 `prefill_chunk()` 依次完成以下动作：

1. 检查上下文预算、终态和本次 chunk 大小。
2. 调用执行器，把成功消费的 prompt 数写入 `prefill_offset`。
3. 最后一段 Prefill 返回 logits 后，采样并提交首个生成词元。

后续 `decode_one()` 读取最后一个生成词元，调用执行器并提交下一次采样结果。只有模型调用成功且 logits 非空时，请求才增加输出；执行器失败会让请求进入 `Failed`，不会伪造占位词元。

| 可观察对象 | Prefill 后 | 一次 Decode 后 |
| --- | --- | --- |
| `prefill_offset` | 等于 prompt 长度 | 不再变化 |
| `generated_size` | 增加 1 | 再增加 1 |
| `phase` | `Decoding` 或 `Finished` | `Decoding` 或 `Finished` |
| 模型 Context | 保存 prompt 的分层 K/V | 追加上一个生成词元的 K/V |
| 教学词元历史 | prompt 加首个输出 | 再追加一个输出 |

Chunked Prefill 只是把第一行拆成多次模型调用；如何决定每次允许消费多少 prompt 词元属于 Chapter08 Scheduling，模型 K/V 的分页存储属于 Chapter07。

## 验证阶段变化

执行 Generation Core 测试：

```bash
cmake -S code/chapter06_generation -B /tmp/cpp-llm-chapter06-core \
  -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/cpp-llm-chapter06-core \
  --target chapter06_generation_test -j2
ctest --test-dir /tmp/cpp-llm-chapter06-core \
  -R '^chapter06_generation_test$' --output-on-failure
```

测试固定三个 prompt 词元和三个输出词元，检查 Prefill 只提交一次、请求从 `Prefilled` 进入 `Finished`，教学词元历史最终包含六个词元；超出上下文预算的请求必须在模型执行前进入 `Rejected`。下一篇把 Prefill、采样和 Decode 连接成单请求循环。
