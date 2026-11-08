# Chapter 05：量化与 CPU/GPU 后端

本目录从教学 INT8 Tensor 进入 GGUF 使用的 Q4_K、Q6_K 和 Q8_0 block，并用同一组输入比较完整反量化、边解码边点积和 AVX2/FMA 三种 CPU 路径。`code/cuda` 同时保留批量反量化加 cuBLAS 的正确性基线和单 token 融合 GEMV。阅读代码前先看 [Chapter05 导读](../../source/chapter05_quantization/intro.md)，理解量化误差、block 布局和仅权重量化路径。

## 代码架构

Chapter01 持有 GGUF 文件缓冲区和模型权重，Chapter05 的 `QuantizedMatrixView` 只借用 packed payload；模型对象必须活得比 Decoder 和所有 Linear 调用更久。Float32 激活由 Chapter02 View 传入，CPU policy 选择已注册 backend，CUDA 则由独立执行上下文管理设备权重和 workspace。跨章适配器先写临时结果，整批成功后才提交到调用方输出。

```text
packed 权重 + Float32 激活
          │
          ├─ 完整反量化 → Float32 权重 → Linear       数值参考
          ├─ scalar fused：解一个 block、立即点积      CPU 主线
          ├─ AVX2/FMA fused                            可选 CPU 加速
          ├─ CUDA 反量化 → workspace → cuBLAS SGEMM   批量正确性基线
          └─ CUDA fused GEMV：解码、点积、归约        单 token 优化路径
```

`QuantizedLinearPolicy` 只选择已经注册且运行机器支持的 CPU backend；请求不可用实现时返回明确 `Status`。

## 阅读顺序与资产

| 顺序 | 文件 | 阅读重点 | 对应验证 |
| --- | --- | --- | --- |
| 1 | `src/tensor.cpp`、`src/dequantize_op.cpp` | INT8、缩放因子和教学反量化路径 | `chapter05_quantization_test` 的 Tensor 与 INT8 用例 |
| 2 | `src/q4k_dequantize.cpp`、`src/q6k_dequantize.cpp`、`src/q8_0_dequantize.cpp` | 三种 GGUF block 如何恢复数值 | 同一测试中的 Q4_K/Q6_K/Q8_0 固定结果 |
| 3 | `src/q4k_kernel.cpp`、`src/q6k_kernel.cpp`、`src/q*_linear.cpp` | reference、边解码边点积和 backend 契约 | reference、scalar fused 与 AVX2 数值对照 |
| 4 | `src/quantized_linear_policy.cpp` | 格式、CPU 能力与已注册 backend 怎样决定执行实现 | policy 缺失、非法类型和可用 AVX2 用例 |
| 5 | `src/quantized_linear_view_adapter.cpp` | Chapter02 rank-1/2 激活怎样进入量化 Linear | `chapter02_tensor_quantized_linear_integration_test` |
| 6 | `examples/q4k_benchmark.cpp` | 生成 baseline、fused 和 SIMD 原始测量 | Chapter09 性能证据流程 |
| 7 | `../cuda` 量化 Linear 资产 | CUDA 批量反量化加 SGEMM 与单 token 融合 GEMV | CUDA 正确性测试、EXP-GPU-02/03 |

阅读一种格式时，先看对应的 `*_dequantize.cpp`，确认 block 内缩放因子和整数位布局；再看 `q*k_kernel.cpp`，理解为什么边解码边点积可以省去完整浮点权重缓冲区；最后进入 policy 和 View adapter，观察执行选择与跨章数据交接。三种格式的 block 尺寸和编码方式不同，各自保留独立解码实现。

## 运行与验证

从仓库根目录执行：

```bash
bash code/chapter05_quantization/scripts/run.sh
```

示例应输出 `quantized ready: true`、反量化值 `0.1`、`2x2` 输出以及 `0.5`、`2.2` 两个固定结果。随后三个 CTest 提供以下证据：

| 层级 | 测试 | 证明内容 |
| --- | --- | --- |
| component | `chapter05_quantization_test` | INT8、Q4_K/Q6_K/Q8_0 解码、Linear、policy、AVX2 和失败路径 |
| integration | `chapter02_tensor_quantized_linear_integration_test` | Chapter02 rank-1/2 View 进入量化 batch backend，失败时保留输出 |
| contract | `cpp_llm_engine_core_test` | 跨章 Tensor、Linear 和量化字节数契约 |
| real_model | `gguf_decoder_multilayer` | TinyLlama 22 层、Context 隔离、embedding-prefill、llama.cpp 数值指标和 greedy 序列 |

真实 TinyLlama 验证需要已下载的 GGUF 和本地 llama.cpp：

```bash
bash code/chapter05_quantization/scripts/run_real_decoder.sh
```

Q4_K/Q6_K 已完成 CPU block reference、scalar fused、AVX2/FMA fused backend，以及 GPU 批量反量化加 SGEMM 和单 token 融合 GEMV。Q8_0 已完成 CPU block reference、scalar fused、统一 policy 和真实 SmolVLM 32 层 Decoder/LM Head 对照；Q8_0 的 AVX2 与 GPU backend 尚未实现。`QuantizedLinearPolicy` 管理 CPU backend，GPU 路径由 `CudaExecutionContext` 编排。

## 扩展入口

新增量化格式时先在 Core 登记 block 布局和字节数校验，再提供独立反量化 reference，最后实现 fused 或 SIMD backend。只有 reference 数值测试通过后才能加入 policy；性能结论由 Chapter09 登记，本章 benchmark 只生成原始测量。

当前代码只实现权重量化。激活量化仍需增加静态校准或动态 scale、激活 dtype contract、低精度 GEMM backend 和 Decoder 误差测试；K/V Cache 量化不属于本目录，由 Chapter07 的 payload 与 Attention 路径扩展。
