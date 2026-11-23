# F11 数值验证与正确性证据

推理计算使用浮点数、量化权重和不同设备规约顺序，同一公式通常不会得到逐位相同的结果。数值验证要先固定参考路径，再选择能解释差异的指标和容差。

## 从参考结果到指标

CPU 的可读实现、PyTorch 或成熟框架都可以提供参考结果。候选实现必须使用相同输入、shape、数据类型和模型状态，再比较以下指标：

| 指标 | 回答的问题 | 适合场景 |
| --- | --- | --- |
| 最大绝对误差 | 最坏元素偏差多大 | Kernel、单层输出 |
| 均方根误差（RMSE） | 整体误差能量多大 | Hidden state、K/V |
| 余弦相似度 | 两个向量方向是否一致 | Logits、量化 Linear |
| Top-K 或贪心词元 | 最终决策是否一致 | 模型层和生成 |

单一指标不能回答所有问题。较高余弦相似度可能掩盖某个元素的大偏差；贪心词元一致也不能证明中间 Tensor 正确。验证应从元素误差逐步扩大到模型决策。

## 容差来自计算路径

容差需要记录数据类型、累加长度、量化格式和规约顺序。FP32 CPU 与 FP16 CUDA 的容差通常不同，经过 20 多层 Decoder 后也不能沿用单个算子的阈值。教程各章负责给出自己的阈值；Foundation 只固定选择方法：

```text
固定输入 -> 参考结果 -> 元素误差 -> 向量相似度 -> 决策一致性
```

出现超差时，先缩小到第一处发生偏差的层或算子，再检查 shape、layout、dtype 和位置编码参数。

## 数值稳定的 Softmax

直接计算 `exp(logit)` 可能溢出。Softmax 对所有 logit 同时平移不敏感，因此先减去最大值：

```text
probability[i] = exp(logit[i] - max(logit)) / sum(exp(logit - max(logit)))
```

这一步既是 Attention 的基础，也是理解在线 Softmax 和分块 Attention 的入口。

## 验证与练习

完成 [`FQ18 Numerical Evidence`](https://github.com/datawhalechina/cpp-llm-engine/blob/main/exercises/foundation/problems/fq18_numerical_evidence/README.md)，实现三种误差指标和稳定 Softmax。随后进入 Chapter03 查看算子对照，进入 Chapter05 查看量化误差怎样影响模型输出。
