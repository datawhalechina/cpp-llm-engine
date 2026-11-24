# Chapter 05：量化与 CPU/GPU 后端

本目录从教学 INT8 张量进入 GGUF 分块量化格式，并用同一组输入比较完整反量化、边解码边点积和可选 SIMD 路径。TinyLlama 使用 Q4_K/Q6_K，SmolVLM 使用 Q8_0，Qwen1.5-MoE 混用 Q3_K/Q4_K/Q5_0/Q5_1/Q5_K，DeepSeek-V2-Lite 还需要 Q2_K 与 IQ4_NL。`code/cuda` 同时保留完整反量化加 cuBLAS 的正确性基线，以及单词元和小批量融合 GEMV。阅读代码前先看 [Chapter05 导读](../../source/chapter05_quantization/intro.md)，理解量化误差、分块布局和仅权重量化路径。

## 代码架构

Chapter01 持有 GGUF 文件缓冲区和模型权重，Chapter05 的 `QuantizedMatrixView` 只借用打包载荷；模型对象必须活得比解码器和所有线性层调用更久。Float32 激活由 Chapter02 视图传入，CPU 策略选择已注册后端，CUDA 则由独立执行上下文管理设备权重和工作区。跨章适配器先写临时结果，整批成功后才提交到调用方输出。

```text
packed 权重 + Float32 激活
          │
          ├─ 完整反量化 → Float32 权重 → Linear       数值参考
          ├─ scalar Linear：边解码边点积或使用块临时值 CPU 主线
          ├─ AVX2/FMA fused                            可选 CPU 加速
          ├─ CUDA 反量化 → workspace → cuBLAS SGEMM   正确性与性能基线
          └─ CUDA fused GEMV：二维网格覆盖单行或批量  Decode 优化路径
```

`QuantizedLinearPolicy` 从已注册且运行机器支持的 CPU 计算后端中选择实现；请求指定不可用实现时返回明确 `Status`。

`chapter05_quantization` 提供通用量化库，x86_64 构建还提供独立的 `chapter05_quantization_avx2` 加速目标。直接构建本章时，`CHAPTER05_BUILD_STANDALONE_TARGETS` 默认开启并生成示例、benchmark 与测试；Model 和多模态等上层模块可以关闭该选项，只链接所需库目标。

### 设计分层

| 层次 | 主要对象 | 职责 |
| --- | --- | --- |
| 公共格式接口 | 核心 `WeightFormat`、`LinearRequest` | 统一格式标识、输入输出形状和后端调用方式 |
| 打包布局 | Core 与 Chapter05 的 block 常量 | CPU/CUDA 共用格式在 Core 登记；CPU 专用的 Q5 布局由 Chapter05 持有 |
| 数值参考 | `q*_dequantize.cpp`、各格式 reference | 独立解释 GGUF 字节，作为融合实现的正确性基线 |
| CPU 执行 | 融合计算后端、`QuantizedLinearPolicy`、可选 AVX2 注册表 | 按格式和显式策略选择已注册实现 |
| CUDA 执行 | `code/cuda` 的量化矩阵、反量化与融合 GEMV | 管理设备载荷、工作区、单词元和小批量计算内核 |
| 模型接入 | `code/model`、`code/multimodal` | 把 GGUF 张量绑定到线性层，向解码器和运行时提供统一调用接口 |

### 格式能力矩阵

| 格式 | 独立解码/reference | CPU 标量 Linear | CPU AVX2/FMA | CUDA 反量化与 fused GEMV | 主要真实模型 |
| --- | --- | --- | --- | --- | --- |
| Q2_K | 是 | 是 | - | 是 | DeepSeek-V2-Lite |
| Q3_K | 是 | 是 | - | 是 | Qwen1.5-MoE |
| Q4_K | 是 | 是 | 是 | 是 | TinyLlama、Qwen1.5-MoE |
| Q5_0 | 是 | 是 | - | - | Qwen1.5-MoE |
| Q5_1 | 是 | 是 | - | - | Qwen1.5-MoE |
| Q5_K | 固定 golden | 是，先解码临时块 | - | - | Qwen1.5-MoE |
| Q6_K | 是 | 是 | 是 | 是 | TinyLlama |
| Q8_0 | 是 | 是 | - | 是 | SmolVLM |
| IQ4_NL | 是 | 是 | - | 是 | DeepSeek-V2-Lite |

表中的 `-` 表示仓库尚未提供该专用后端。Q3_K、Q4_K、Q5_K、Q6_K 和 Q8_0 的 CPU 标量后端还提供共享权重解码的批量实现；其他格式通过公共逐行批量接口执行。Q5_K 在线性层内部解码完整临时块，并由固定结果和真实 MoE 对照验证。扩展融合或 SIMD 后端时，先增加独立解码参考结果。

## 阅读顺序与资产

| 顺序 | 文件 | 阅读重点 | 对应验证 |
| --- | --- | --- | --- |
| 1 | `src/tensor.cpp`、`src/dequantize_op.cpp` | INT8、缩放因子和教学反量化路径 | `chapter05_quantization_test` 的 Tensor 与 INT8 用例 |
| 2 | `src/q*_dequantize.cpp`、`src/q5k_linear.cpp` | 各 GGUF block 如何恢复数值 | 九种 GGUF 格式的固定结果 |
| 3 | `src/q4k_kernel.cpp`、`src/q6k_kernel.cpp`、`src/q*_linear.cpp` | 参考实现、标量线性层和计算后端接口 | 参考、标量与 AVX2 数值对照 |
| 4 | `src/quantized_linear_policy.cpp` | 格式、CPU 能力与已注册 backend 怎样决定执行实现 | policy 缺失、非法类型和可用 AVX2 用例 |
| 5 | `src/quantized_linear_view_adapter.cpp` | Chapter02 rank-1/2 激活怎样进入量化 Linear | `chapter02_tensor_quantized_linear_integration_test` |
| 6 | `examples/q4k_benchmark.cpp` | 生成 baseline、fused 和 SIMD 原始测量 | Chapter09 性能证据流程 |
| 7 | `../cuda` 量化 Linear 资产 | CUDA 完整反量化基线与单 token/批量融合 GEMV | CUDA 正确性测试、EXP-GPU-02/03 与双请求 A/B |

阅读一种格式时，先看对应的反量化、reference 或固定 golden，确认 block 内缩放因子、整数位或码表布局；再看标量 Linear，比较边解码边点积与块临时值两种组织；最后进入 policy 和 View adapter，观察执行选择与跨章数据交接。九种格式的 block 尺寸和编码方式不同，各自保留独立解码规则。

## 运行与验证

从仓库根目录执行：

```bash
bash code/chapter05_quantization/scripts/run.sh
```

示例应输出 `quantized ready: true`、反量化值 `0.1`、`2x2` 输出以及 `0.5`、`2.2` 两个固定结果。随后四个 CTest 提供以下证据：

| 层级 | 测试 | 证明内容 |
| --- | --- | --- |
| component | `chapter05_quantization_test`、`chapter05_deepseek_quantization_test` | INT8、九种 GGUF 格式的解码、Linear、policy、AVX2 和失败路径 |
| integration | `chapter02_tensor_quantized_linear_integration_test` | Chapter02 rank-1/2 View 进入量化 batch backend，失败时保留输出 |
| 公共接口 | `cpp_llm_engine_core_test` | 跨章张量、线性层和量化字节数检查 |
| real_model | `gguf_decoder_multilayer` | TinyLlama 22 层、Context 隔离、embedding-prefill、llama.cpp 数值指标和 greedy 序列 |

真实 TinyLlama 验证需要已下载的 GGUF 和本地 llama.cpp：

```bash
bash code/chapter05_quantization/scripts/run_real_decoder.sh
```

Q2_K/Q3_K/IQ4_NL 提供 CPU 分块参考、标量融合，以及 CUDA 完整反量化和单词元/批量融合 GEMV，供 DeepSeek-V2-Lite 使用。Q3_K/Q5_1/Q5_K 支撑 Qwen1.5-MoE 专家权重，其中 Q3_K/Q5_K 的 CPU 后端可让多个输入行共享分块解码。Q4_K/Q6_K 提供 CPU 标量与 AVX2/FMA 融合后端，以及 GPU 单词元/批量融合 GEMV。Q8_0 通过 SmolVLM 解码器与 LM Head 对照。`QuantizedLinearPolicy` 管理 CPU 后端，GPU 路径由 `CudaExecutionContext` 编排。

## 扩展入口

新增量化格式时先登记 `WeightFormat`。需要由 CPU 与 CUDA 共用的布局在 Core 增加 block 字节数校验；仅由 Chapter05 使用的布局可以留在格式模块。随后提供独立反量化或等价 reference，再实现 fused 或 SIMD backend。reference 数值测试通过后才能加入 policy；性能结论由 Chapter09 登记，本章 benchmark 只生成原始测量。

代码中的“混合量化”表示同一个 GGUF 模型中的不同张量采用不同权重格式，格式由模型文件确定。激活量化需要增加静态校准或动态缩放、激活数据类型接口、低精度 GEMM 后端和解码器误差测试；K/V Cache 量化则需要扩展 Chapter07 的缓存载荷与注意力路径。
