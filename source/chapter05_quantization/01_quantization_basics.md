# 05.1 用整数近似浮点权重

真实模型的线性层包含大量浮点权重。仅权重量化用较少位数保存这些权重，计算时再按格式恢复近似值；输入激活和输出仍保持浮点表示。

## 缩放、舍入与饱和

最小的对称 INT8 例子使用一个缩放因子（`scale`）：

```text
q = clamp(round(x / scale), -127, 127)
x_hat = q * scale
```

`x` 是原始浮点值，`q` 是保存的整数，`x_hat` 是反量化结果。舍入会丢失低位信息，超出整数范围的值会被饱和到边界，因此通常有 `x_hat != x`。

以 `scale=0.1` 为例：

| 浮点值 `x` | 整数 `q` | 还原值 `x_hat` | 绝对误差 |
| ---: | ---: | ---: | ---: |
| `0.26` | `3` | `0.3` | `0.04` |
| `-0.14` | `-1` | `-0.1` | `0.04` |
| `20.0` | `127` | `12.7` | `7.3` |

第三行显示缩放因子过小时会发生饱和。真实格式会按行、通道或分组保存多个缩放参数，以便更贴近局部数值范围。

## 从教学 INT8 到 GGUF 分块格式

Chapter05 先用“一个整数数组加一个缩放因子”固定公式，再处理 TinyLlama GGUF 中的 Q4_K、Q6_K 和 Q8_0。后者不是普通整数数组，而是由固定大小的 block 共同保存缩放参数、最小值和 packed quant。

| 表示 | 每个数的可见形式 | 计算前必须知道什么 | 本章用途 |
| --- | --- | --- | --- |
| 教学 INT8 | `int8_t` | shape、单个 scale | 解释量化与反量化公式 |
| Q8_0 | 8-bit quant block | block scale 与布局 | 验证简单分块量化路径 |
| Q4_K / Q6_K | packed low-bit super-block | block 字节布局、分组参数和 shape | 执行真实 TinyLlama 权重 |

量化格式决定怎样读取字节，Linear 契约决定输入输出 shape，backend 决定由标量、SIMD 还是 CUDA 完成计算。三者分开后，新增执行路径不需要改变 Decoder 的调用方式。

## 验证与衔接

先手算表中的 `q`、`x_hat` 和误差，再运行：

```bash
bash code/chapter05_quantization/scripts/run.sh
```

`chapter05_quantization_demo` 会输出教学 INT8 的反量化值和 Linear 结果。理解误差来源后，进入 [`05.2 表达量化数据`](./02_extend_tensor.md)，观察 dtype、存储和 scale 怎样进入内存对象。
