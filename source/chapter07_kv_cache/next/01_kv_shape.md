# 07.1 从 Attention 看 K/V 的形状

Chapter06 已经固定单请求的 Prefill 与 Decode 顺序。模型要避免每一步重新计算全部历史，就需要保存每层 Attention 产生的 key 和 value。理解缓存之前先固定逻辑 shape，否则 Block、Page 和物理地址都没有明确的映射对象。

## 一层需要保存什么

设当前解码层使用 `Hkv` 个 KV head，每个 head 的宽度是 `Dh`，已经提交 `S` 个词元。一层的逻辑 K/V 形状都是：

```text
K: [S, Hkv, Dh]
V: [S, Hkv, Dh]
```

Prefill 一次产生多个位置，Decode 每次只产生一个新位置。提交后，序列维度从 `S` 增长到 `S+1`，head 数和 head width 不变。

| 阶段 | 本次新 K/V | 提交后的缓存 | 当前 query 读取范围 |
| --- | --- | --- | --- |
| Prefill `S` 个词元 | `[S, Hkv, Dh]` | `[S, Hkv, Dh]` | 每个位置的因果前缀 |
| 第一次 Decode | `[1, Hkv, Dh]` | `[S+1, Hkv, Dh]` | 位置 `0..S` |
| 第 `g` 次 Decode | `[1, Hkv, Dh]` | `[S+g, Hkv, Dh]` | 位置 `0..S+g-1` |

完整 Attention 公式可查阅[公式表 F-KV-01](/reference/formulas#f-kv-01)。缓存保存的是投影并应用位置编码后的 K/V 数值，不保存 query，也不保存最终 Attention 输出。

## GQA 怎样共享 KV head

TinyLlama 使用分组查询注意力（GQA）：query head 数可以大于 KV head 数，多个 query head 共享同一个 KV head。若 `Hq = 4`、`Hkv = 2`，映射关系是：

| Query head | 读取的 KV head |
| --- | --- |
| 0、1 | 0 |
| 2、3 | 1 |

因此缓存宽度由 `Hkv * Dh` 决定，而不是 `Hq * Dh`。分页后每个词元槽位仍需容纳一整行 `[Hkv, Dh]` 的 key 和 value；Page 只改变物理存放位置，不改变 Attention 看见的逻辑 shape。

## 验证连续 Attention

Chapter04 的测试已经覆盖连续 K/V 的追加、容量拒绝和 Attention 读取：

```bash
cmake -S code/chapter04_attention -B /tmp/cpp-llm-chapter07-shape \
  -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/cpp-llm-chapter07-shape \
  --target chapter04_attention_test -j2
ctest --test-dir /tmp/cpp-llm-chapter07-shape \
  -R '^chapter04_attention_test$' --output-on-failure
```

通过条件包括 K/V 按 `[tokens, heads, head_dim]` 追加、超出 capacity 时拒绝写入、Decode 读取正确历史范围。下一篇把这组逻辑 shape 放进连续 storage。
