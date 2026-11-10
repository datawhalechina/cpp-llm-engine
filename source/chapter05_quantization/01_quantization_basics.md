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

## 先区分量化对象

“INT8”或“INT4”只说明数值表示范围，还没有说明量化了什么对象。推理系统会分别决定权重、激活和 K/V Cache 的表示方式；它们具有不同生命周期，也需要不同 kernel 与误差验证。

| 路线 | 被量化的对象 | 主要收益与代价 | 本教程状态 |
| --- | --- | --- | --- |
| 仅权重量化 | 模型权重 | 降低模型大小和权重读取带宽；激活计算仍是浮点 | Chapter05 主线，已实现 |
| 权重与激活量化 | 权重和中间激活 | 可能继续降低计算与带宽，但需要校准、动态 scale 或专用矩阵 kernel | 只作成熟框架对照 |
| K/V Cache 量化 | 请求运行时产生的 key/value | 降低长上下文缓存容量；会影响 Attention 读取和逐步误差 | Chapter07 扩展项，未实现 |
| 混合量化 | 不同 Tensor 使用不同位宽或格式 | 在模型大小、敏感层精度和 kernel 支持之间折中 | TinyLlama `Q4_K_M` 作为真实资产使用 |

## 从教学 INT8 到 GGUF 分块格式

Chapter05 先用“一个整数数组加一个缩放因子”固定公式，再处理真实 GGUF 中的 Q2_K、Q3_K、Q4_K、Q5_0、Q5_1、Q5_K、Q6_K、Q8_0 和 IQ4_NL。后者不是普通整数数组，而是由固定大小的 block 共同保存缩放参数、最小值、packed quant 或码表索引。

| 表示 | 每个数的可见形式 | 计算前必须知道什么 | 本章用途 |
| --- | --- | --- | --- |
| 教学 INT8 | `int8_t` | shape、单个 scale | 解释量化与反量化公式 |
| Q8_0 | 8-bit quant block | block scale 与布局 | 验证简单分块量化路径 |
| Q2_K | packed 2-bit super-block | 2-bit quant、4-bit 分组 scale/minimum | DeepSeek-V2-Lite 的 Attention 与稠密 FFN 权重 |
| Q5_0 | 有符号 5-bit quant block | block scale、低 4 bit 与第五位表 | Qwen1.5-MoE 部分层的专家下投影 |
| Q5_1 | 5-bit quant block | block scale、minimum 与高位表 | Qwen1.5-MoE 专家下投影 |
| Q3_K | packed low-bit super-block | 3-bit quant 与 6-bit 分组 scale | Qwen1.5-MoE 专家 gate/up |
| Q4_K / Q6_K | packed low-bit super-block | block 字节布局、分组参数和 shape | 执行真实 TinyLlama 权重 |
| IQ4_NL | 4-bit codebook block | block scale 与 16 项非线性码表 | DeepSeek-V2-Lite 稠密层 down projection |

量化格式决定怎样读取字节，Linear 契约决定输入输出 shape，backend 决定由标量、SIMD 还是 CUDA 完成计算。三者分开后，新增执行路径不需要改变 Decoder 的调用方式。

## Q4_K、Q6_K 与 Q8_0 为什么并存

位数越低通常越节省存储和内存带宽，但误差、解码复杂度和可用 kernel 也会变化。模型文件可以为不同 Tensor 选择不同格式；例如 `Q4_K_M` 是一种混合量化模型方案，不表示文件中的所有 Tensor 都是 Q4_K。Loader 按每个 Tensor 的 GGML type 选择解码器，Runtime 不会在每个请求中重新决定权重格式。

| 格式 | 256 个权重的存储量 | 主要特点 | 本教程中的作用 |
| --- | ---: | --- | --- |
| Q2_K | 84 字节，约 2.63 bit/weight | 2-bit 值配合 16 组 scale/minimum，在低位宽下补偿局部范围 | DeepSeek-V2-Lite Attention 和稠密 FFN |
| Q4_K | 144 字节，约 4.5 bit/weight | 存储较小，同时使用分组 scale 与 minimum 控制误差 | TinyLlama 多数 Q4_K 权重、CPU/CUDA 融合执行主线 |
| Q6_K | 210 字节，约 6.56 bit/weight | 比 Q4_K 保存更多量化位，通常用于更敏感或需要更高精度的 Tensor | TinyLlama FFN down 等真实权重验证 |
| Q8_0 | 272 字节，约 8.5 bit/weight | 对称结构简单，每 32 个权重共享一个 scale，但存储更大 | 简单 block 参考和 SmolVLM 真实多模态权重 |
| Q3_K | 110 字节，约 3.44 bit/weight | 256 个值共享总 scale，分组 scale 补偿低位精度 | Qwen1.5-MoE 打包 gate/up 权重 |
| Q5_0 | 22 字节，5.5 bit/weight | 32 个值共享 scale，5-bit quant 恢复后减 16 | Qwen1.5-MoE 打包 down 权重 |
| Q5_1 | 24 字节，6 bit/weight | 32 个值共享 scale 与 minimum，5-bit quant 由低四位和高位表组成 | Qwen1.5-MoE 打包 down 权重 |
| Q5_K | 176 字节，5.5 bit/weight | 256 个值按八组共享 scale 与 minimum，并单独保存第五位 | Qwen1.5-MoE 共享专家 down 权重 |
| IQ4_NL | 每 32 个权重 18 字节，4.5 bit/weight | 4-bit 索引不按等距整数解释，而是查 16 项非线性码表后乘 block scale | DeepSeek-V2-Lite 稠密 FFN down projection |

Q2_K 把 256 个权重分成更小的局部分组，每组从 4-bit 元数据取得 scale 与 minimum，再还原 2-bit 值。IQ4_NL 则把每个 4-bit 值作为码表下标；码表在零点附近更密、在两端更疏，因此“4 bit”不等于线性映射到 `0..15`。两种格式都必须按 GGUF block 布局解码，不能只凭位宽解释字节。

这里的 `K` 表示 ggml 的 K-quant 分块格式族，与 Attention 的 K/V Cache 没有关系。格式选择需要同时比较模型精度、文件大小、内存带宽和目标设备 kernel，不能只按位数给出统一优劣顺序。

## 验证与衔接

先手算表中的 `q`、`x_hat` 和误差，再运行：

```bash
bash code/chapter05_quantization/scripts/run.sh
```

`chapter05_quantization_demo` 会输出教学 INT8 的反量化值和 Linear 结果。理解误差来源后，进入 [`05.2 表达量化数据`](./02_extend_tensor.md)，观察 dtype、存储和 scale 怎样进入内存对象。
