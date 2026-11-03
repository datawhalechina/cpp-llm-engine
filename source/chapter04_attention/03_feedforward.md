# 04.3 前馈网络

注意力在词元之间传递信息，FFN 则独立变换每个词元的特征。教学实现先用两个线性层和 ReLU 建立可运行结构，真实 TinyLlama 再用门控、升维、降维投影和 SwiGLU 对齐模型权重。

## 从隐藏维升维再降维

设输入为 `[T,D]`，中间维度为 `F`。标准两层 FFN 的形状流程是：

```text
输入 [T,D]
  -> fc1[D,F]
  -> hidden [T,F]
  -> ReLU
  -> activated [T,F]
  -> fc2[F,D]
  -> 输出 [T,D]
```

中间维度扩大了单个词元可使用的特征空间，但不会改变词元数。`FeedForward(4,8)` 因此把 `[2,4]` 暂时变为 `[2,8]`，再恢复为 `[2,4]`，以便与残差分支逐元素相加。

<a href="../_code/chapter04_attention/src/feed_forward.cpp"><code>feed_forward.cpp</code></a> 直接表达这三个步骤：

```cpp
Tensor hidden = fc1_.forward(input);
Tensor activated = apply_activation(hidden, ActivationKind::Relu);
return fc2_.forward(activated);
```

两个线性层复用 <a href="../_code/chapter04_attention/src/linear_layer.cpp"><code>linear_layer.cpp</code></a>。`forward()` 保留 Chapter04 的教学张量接口；`try_forward()` 接收公共只读输入视图和可写输出视图，因此 Chapter02 张量可以直接进入同一线性计算。两条入口都要求 CPU Float32、二维连续布局和匹配的输入输出形状，不会替调用者隐式连续化。

线性层、LayerNorm、ReLU、残差相加和完整 FFN 均提供公共视图接口。Chapter04 教学张量仍负责旧入口和中间结果的存储所有权，公共视图只负责跨章传递布局与数据引用。

## 从教学 FFN 到真实 SwiGLU

TinyLlama 的 FFN 不是简单地把 ReLU 换成 SiLU。它使用三组权重并在两个升维分支之间执行逐元素门控：

```text
gate = gate_proj(x)
up = up_proj(x)
hidden = SiLU(gate) * up
output = down_proj(hidden)
```

| 对照项 | 教学 `FeedForward` | TinyLlama 真实 FFN |
| --- | --- | --- |
| 升维投影 | `fc1` | `gate_proj`、`up_proj` |
| 激活 | ReLU | SiLU |
| 门控乘法 | 无 | `SiLU(gate) * up` |
| 降维投影 | `fc2` | `down_proj` |
| 权重数据 | 教学 Float32 参数 | GGUF Q4_K/Q6_K 权重 |

两条路径保留的共同约束是：词元数不变，最终隐藏维恢复为 `D`，输出才能进入残差加法。真实路径位于 `RealLlamaCpuContext`，量化线性层复用 Chapter05 计算后端；教学 `FeedForward` 保持小尺寸和可读性。

在解码器中，注意力和 FFN 各自位于一条残差支路：

```text
x = x + Attention(Norm(x))
x = x + FFN(Norm(x))
```

因此 FFN 的输入输出形状是否一致不是实现细节，而是残差连接成立的前提。

## 验证与衔接

从仓库根目录运行：

```bash
bash code/chapter04_attention/scripts/run.sh
```

`chapter04_attention_test` 构造 `FeedForward(4,8)`，输入 `[2,4]` 后检查输出仍为 `[2,4]`。跨章测试还把 Chapter02 的 `[1,2]` 张量送入 `LinearLayer(2,3)` 和 `FeedForward(2,3)`，分别验证 `[0.05,0.10,0.15]` 与 `[0.007,0.014]`，并确认错误输出形状不会覆盖已有数据。

下一节将把两条残差支路、两个 LayerNorm、注意力和 FFN 组装成一个完整的教学 `DecoderLayer`。
