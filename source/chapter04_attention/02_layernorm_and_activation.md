# 04.2 层归一化与激活函数

注意力负责聚合词元之间的信息，解码器还需要归一化稳定每个词元的特征尺度，并用激活函数引入非线性。教学路径用 LayerNorm 和 ReLU 展示基本计算，TinyLlama 则按模型权重执行 RMSNorm 和 SwiGLU。

## 每个词元怎样完成归一化

输入 `[T,D]` 中的每一行表示一个词元。LayerNorm 只在该行的 `D` 个特征上计算均值和方差，不混合不同词元，也不改变形状：

```text
mean = sum(x_i) / D
variance = sum((x_i - mean)^2) / D
y_i = (x_i - mean) / sqrt(variance + epsilon) * gamma_i + beta_i
```

| 对象 | 当前值或形状 | 作用 |
| --- | --- | --- |
| 输入与输出 | `[T,D]` | 词元数和隐藏维度保持不变 |
| `mean`、`variance` | 每个词元各一个标量 | 只统计当前词元的特征 |
| `epsilon` | 默认 `1e-5` | 防止方差接近零时除零 |
| `gamma`、`beta` | `[D]` | 当前构造函数分别初始化为 `1` 和 `0` |

LayerNorm 在推理时仍使用当前输入计算统计量，不读取 BatchNorm 式的 running mean/variance。<a href="https://github.com/datawhalechina/cpp-llm-engine/blob/main/code/chapter04_attention/src/layer_norm.cpp"><code>layer_norm.cpp</code></a> 中的三段循环依次计算均值、方差和输出，正好对应上面的公式。

`forward()` 保留 Chapter04 教学张量入口，`try_forward()` 接收公共只读输入视图和可写输出视图。两条入口都要求 CPU Float32、二维连续布局、输入列数等于 `dim`，并保持输入输出形状一致。计算先写入临时结果，全部成功后再提交到输出，因此错误形状不会破坏调用方已有数据。

## 教学实现与 TinyLlama 的差异

教学 `FeedForward` 使用 ReLU：

```text
ReLU(x) = max(0, x)
```

真实 TinyLlama 使用 RMSNorm 和 SwiGLU。RMSNorm 不减均值，而是按均方根缩放；SwiGLU 需要 gate、up、down 三组投影：

```text
RMSNorm(x) = x / sqrt(mean(x^2) + epsilon) * weight
SwiGLU(x) = down(SiLU(gate(x)) * up(x))
SiLU(x) = x / (1 + exp(-x))
```

| 对照项 | 教学解码器 | TinyLlama 真实解码器 |
| --- | --- | --- |
| 归一化 | LayerNorm | RMSNorm |
| 是否减均值 | 是 | 否 |
| 可学习参数 | `gamma`、`beta` | 缩放权重 |
| FFN 激活 | ReLU | SiLU 门控 |
| 作用 | 建立可读组合结构 | 匹配真实模型配置和权重 |

`apply_activation()` 位于 <a href="https://github.com/datawhalechina/cpp-llm-engine/blob/main/code/chapter04_attention/src/feed_forward.cpp"><code>feed_forward.cpp</code></a>，当前枚举包含教学路径实际执行的 ReLU。TinyLlama 的 RMSNorm 和 SwiGLU 位于真实 CPU 解码器路径，并读取对应 GGUF 权重。

## 验证与衔接

运行：

```bash
bash code/chapter04_attention/scripts/run.sh
```

`chapter04_attention_test` 使用 `[2,4]` 输入验证 `LayerNorm(4)` 已就绪，输出仍为 `[2,4]`；跨章测试将 Chapter02 的 `[1,2]` 张量归一化为约 `[-0.99998,0.99998]`，并检查错误输出形状不覆盖已有数据。同一测试随后把归一化对象交给完整 `DecoderLayer`。检查实现时应确认每一行独立统计、`epsilon` 位于平方根内部、ReLU 把负值变为零且保留正值。

下一节会把线性层、ReLU 和第二个线性层组合成保持 `[T,D]` 输入输出形状的 FFN。
