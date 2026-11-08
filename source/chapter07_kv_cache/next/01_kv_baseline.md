# 07.1 从 K/V 形状到连续缓存

Chapter06 已经固定 Prefill 与 Decode 的执行顺序。模型若不保存历史 key 和 value，每次 Decode 都要重新计算整个前缀。本节先确定缓存保存什么，再用连续存储建立分页实现的数值基线。

## Attention 保存什么

设一层使用 `Hkv` 个 K/V 头，每个头宽 `Dh`，已经提交 `S` 个词元。key 和 value 的逻辑形状相同：

```text
K: [S, Hkv, Dh]
V: [S, Hkv, Dh]
```

Prefill 一次产生多个位置；Decode 每次追加一个位置。缓存保存的是投影并应用位置编码后的 K/V，不保存 query 和 Attention 输出。完整公式见[公式表 F-KV-01](/reference/formulas#f-kv-01)。

TinyLlama 使用分组查询注意力（GQA）。当 `Hq = 4`、`Hkv = 2` 时，query head 0、1 共用 KV head 0，query head 2、3 共用 KV head 1。因此每个词元占用的缓存宽度是 `Hkv * Dh`，而不是 `Hq * Dh`。

| 阶段 | 本次新 K/V | 提交后的缓存 | 当前 query 读取范围 |
| --- | --- | --- | --- |
| Prefill `S` 个词元 | `[S, Hkv, Dh]` | `[S, Hkv, Dh]` | 每个位置的因果前缀 |
| 第一次 Decode | `[1, Hkv, Dh]` | `[S+1, Hkv, Dh]` | 位置 `0..S` |
| 第 `g` 次 Decode | `[1, Hkv, Dh]` | `[S+g, Hkv, Dh]` | 位置 `0..S+g-1` |

## 连续缓存怎样寻址

Chapter04 的 `KvCache` 按 `[token, kv_head, head_dimension]` 保存数据。词元位置 `t`、K/V 头 `h`、头内维度 `d` 的元素偏移为：

```text
offset = (t * Hkv + h) * Dh + d
```

key 和 value 分别使用一块连续 `float` 存储。`append` 只在容量足够时复制新行并增加有效长度；`clear` 将有效长度重置为 0，不必清空全部内存，因为 Attention 只读取有效范围。实现见 <a href="/_code/chapter04_attention/src/kv_cache.cpp"><code>kv_cache.cpp</code></a>。

真实模型中，`RealLlamaCpuModel` 保存多请求共享的只读权重，`RealLlamaCpuContext` 保存单请求的位置与每层 K/V。连续存储便于核对地址和数值，但不同请求的闲置容量无法直接互相借用，这正是分页缓存要解决的问题。

| 对象 | 生命周期 | 保存内容 |
| --- | --- | --- |
| `RealLlamaCpuModel` | 多请求共享 | 量化权重和模型配置 |
| `RealLlamaCpuContext` | 单请求独占 | position、每层 K/V、执行统计 |
| `Request` | 单请求独占 | prompt、输出词元和终态 |

## 验证连续基线

先验证连续 K/V 的追加、容量拒绝与 Attention 读取；真实模型构建目录已存在时，再检查两个请求不会共享 Context。

```bash
cmake -S code/chapter04_attention -B /tmp/cpp-llm-chapter07-baseline \
  -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/cpp-llm-chapter07-baseline \
  --target chapter04_attention_test -j2
ctest --test-dir /tmp/cpp-llm-chapter07-baseline \
  -R '^chapter04_attention_test$' --output-on-failure
```
