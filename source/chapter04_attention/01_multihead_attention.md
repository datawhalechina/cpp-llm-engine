# 04.1 多头注意力

Chapter03 已经固定矩阵乘法的形状、布局和计算后端分层；本节把同样的计算语义放进第一个模型结构：输入先生成 Q、K、V，再按注意力头计算相关性并聚合 V。Chapter04 的教学线性层保留本地可读循环，注意力通过独立计算后端执行。学习时重点追踪每一步形状，而不是先记住公式名称。

教学 `MultiHeadAttention` 接收连续二维输入 `[T,D]`；跨章张量适配器接收 `[T,H,Dh]` 或 `[B,T,H,Dh]`。两条路径最终都把连续缓冲区交给同一个 `AttentionRequest`。

## Q、K、V 怎样产生输出

设词元数为 `T`、隐藏维度为 `D`、注意力头数为 `H`，每个头的宽度为 `Dh = D / H`。`D` 必须能被 `H` 整除。

```text
输入 X[T,D]
  -> Q/K/V 线性投影
  -> Q/K/V[T,H,Dh]
  -> score[T,H,T] = Q @ K^T / sqrt(Dh)
  -> causal mask
  -> softmax
  -> head output[T,H,Dh] = weight @ V
  -> 合并为 [T,D]
  -> 输出投影
```

生成模型使用因果掩码。第 `t` 个查询只能读取位置 `0..t` 的键和值，未来位置在 softmax 前写成负无穷：

```text
score = (Q @ K^T) / sqrt(Dh)
score = score + causal_mask
weight = softmax(score)
output = weight @ V
```

当前代码维护三种输入形态，它们表达同一套注意力语义：

| 路径 | 输入布局 | 当前用途 |
| --- | --- | --- |
| 教学 `MultiHeadAttention` | `[T,D]` | 串起 Q/K/V 投影、CPU 注意力和输出投影 |
| Chapter02 张量适配器 | `[T,H,Dh]`、`[B,T,H,Dh]` | 验证维数、连续布局和批次隔离 |
| TinyLlama CPU 执行上下文 | 词元隐藏向量与分层 KV | 执行 GQA、RoPE、KV Cache 和 22 个解码器块 |

通用资料常写 `[B,H,T,Dh]`。本教程的跨章适配器使用词元优先（token-major）布局 `[B,T,H,Dh]`，让同一个词元的所有注意力头连续存放。阅读下图时从左到右追踪形状，再在下半部分比较预填充与逐词元解码的请求差异。

![多头注意力的形状、因果掩码与预填充/解码数据流](../assets/diagrams/chapter04-attention-dataflow.svg)

## 从算法镜像进入 C++

下面的 Python 只镜像 CPU 参考实现的循环顺序，用来先看清算法，不形成第二套运行时：

```python
def attention_reference(query, key, value, causal, offset):
    output = zeros_like(query)
    scale = 1.0 / sqrt(query.head_dim)

    for query_token in range(query.token_count):
        query_position = offset + query_token
        for head in range(query.head_count):
            scores = []
            for key_token in range(key.token_count):
                if causal and key_token > query_position:
                    scores.append(float("-inf"))
                else:
                    scores.append(dot(query[query_token, head],
                                      key[key_token, head]) * scale)

            probabilities = stable_softmax(scores)
            for key_token, probability in enumerate(probabilities):
                output[query_token, head] += (
                    probability * value[key_token, head])
    return output
```

对照源码时按下表逐项定位：

| 算法动作 | C++ 对象或位置 | 观察重点 |
| --- | --- | --- |
| Q/K/V 与输出投影 | <a href="../_code/chapter04_attention/src/attention.cpp"><code>attention.cpp</code></a> | `[T,D]` 怎样进入和离开计算后端 |
| 查询的绝对位置 | `query_position_offset` | 解码查询怎样读取历史 K/V |
| 因果判断 | `key_token > query_position` | 未来词元不进入 softmax |
| 缩放点积 | 注意力 CPU 内核 | 每个注意力头独立计算，默认缩放因子为 `1/sqrt(Dh)` |
| 稳定 softmax | 最大值、指数、分母三段循环 | 大分数不直接进入 `exp` |
| 3D/4D 视图适配 | <a href="../_code/chapter04_attention/src/tensor_attention_adapter.cpp"><code>tensor_attention_adapter.cpp</code></a> | 批次指针偏移与连续性检查 |

适配器不隐式复制非连续视图。Chapter02 的转置或切片如果破坏连续布局，调用者必须先执行 `contiguous()`，复制成本因此保持可见。`try_compute_tensor_attention_into()` 接收调用方提供的可写输出视图；它先完成全部批次计算，成功后统一提交，因此形状校验或计算后端失败不会覆盖已有输出。返回 Chapter02 张量的旧入口继续作为兼容外壳。

## 从单次 Attention 进入 KV Cache

Prefill 可以同时产生多个位置的 Q/K/V；decode 每次只产生一个新位置。连续缓存把新 K/V 追加到 `[S,Hkv,Dh]` 尾部，再让当前 query 读取更新后的 `S+1` 个位置。分页缓存改变存储和寻址方式，但不改变这套注意力语义。

Chapter04 的 `KvCache` 用固定容量连续数组验证追加、容量和清空；真实 TinyLlama Context 为 22 层分别保存连续 K/V，并在每次 decode 后推进 position。Block 分配和逻辑位置映射由 Chapter07 负责，Chapter10 在请求终态触发释放。

| 观察点 | 连续 CPU 路径 | 分页路径目标 |
| --- | --- | --- |
| 写入新 K/V | 追加到每层连续数组尾部 | 写入页表给出的 physical block/slot |
| 读取历史 | 按连续位置 `0..S` 遍历 | 按 block table 定位 `0..S` |
| 序列增长 | `size: S -> S+1` | committed tokens: `S -> S+1` |
| 计算结果 | 当前 query 聚合历史 value | 与连续路径 logits 一致 |

## 验证与衔接

从仓库根目录运行：

```bash
bash code/chapter04_attention/scripts/run.sh
```

测试提供四组可观察证据：因果预填充的首个输出只能读取首个 V；`query_position_offset=1` 的逐词元解码可以读取两项历史 V，并得到 `51`；4D 输入中的两个批次分别保持数值 `3` 和 `7`；非连续视图在显式连续化前被拒绝。

验证还应看到 `MultiHeadAttention(4,2)` 保持输入输出形状 `[2,4]`，计算后端类型为 CPU。下一节将补上解码器中围绕注意力和 FFN 使用的归一化与激活函数。
