# Chapter 05：量化与 CPU/GPU 后端

本目录从教学 INT8 Tensor 进入 GGUF 使用的 Q4_K、Q6_K 和 Q8_0 block，并用同一组输入比较完整反量化、边解码边点积和 AVX2/FMA 三种 CPU 路径。GPU 非融合基线位于 `code/cuda`。原理和图解见 [Chapter05 导读](../../source/chapter05_quantization/intro.md)。

## 代码架构

```text
packed 权重 + Float32 激活
          │
          ├─ 完整反量化 → Float32 权重 → Linear       数值参考
          ├─ scalar fused：解一个 block、立即点积      CPU 主线
          ├─ AVX2/FMA fused                            可选 CPU 加速
          └─ CUDA 反量化 → workspace → cuBLAS SGEMM   GPU 正确性基线
```

`QuantizedLinearPolicy` 只选择已经注册且运行机器支持的 CPU backend，不静默回退。跨章 View 入口先把结果写入临时缓冲区，backend 成功后才提交到调用方输出，因此错误不会留下部分结果。

## 阅读顺序与资产

| 资产 | 作用 |
| --- | --- |
| `src/dequantize_op.cpp` | 通用反量化算子 |
| `src/q4k_dequantize.cpp`、`src/q6k_dequantize.cpp`、`src/q8_0_dequantize.cpp` | Q4_K/Q6_K/Q8_0 block 解码 |
| `src/linear_layer.cpp`、`src/quantized_linear.cpp` | 量化 Linear 路径 |
| `src/quantized_linear_view_adapter.cpp` | packed 权重与 rank-1/2 Float32 激活 View 的跨章适配 |
| `src/q4k_kernel.cpp` | Q4_K fused CPU kernel |
| `src/q4k_linear.cpp` | Q4_K scalar reference 和 backend 适配 |
| `src/q4k_linear_avx2.cpp` | Q4_K fused AVX2/FMA 解码与点积 backend |
| `src/q6k_linear_avx2.cpp` | Q6_K fused AVX2/FMA 解码与点积 backend |
| `src/q8_0_linear.cpp` | Q8_0 reference、scalar fused 与 backend 适配 |
| `include/chapter05_quantization/quantized_linear_policy.h`、`src/quantized_linear_policy.cpp` | Q4_K/Q6_K/Q8_0 统一 backend policy、registry 与 dispatcher |
| `tests/chapter05_quantization_test.cpp` | dtype、多 block、误差、policy 和 backend 测试 |
| `tests/chapter02_tensor_quantized_linear_integration_test.cpp` | Chapter02 激活进入 Q4_K/Q6_K/Q8_0 backend 的跨章测试 |
| `examples/q4k_benchmark.cpp` | Q4_K baseline/fused/SIMD benchmark |
| `scripts/run.sh` | 隔离构建、CTest 和 demo 入口 |
| Chapter 01/06 与多模态 examples | TinyLlama Q4_K/Q6_K、SmolVLM Q8_0 跨章验证 |
| `../cuda` Quantized Linear 资产 | `CudaQuantizedMatrix -> CUDA dequantize -> Float32 Workspace -> cuBLAS SGEMM` |

先选一种格式阅读 `*_dequantize.cpp`，弄清 block 内 scale 和 quant 的布局；再对照 `q*k_kernel.cpp` 理解 fused 为什么省去完整权重缓冲区；最后阅读 `quantized_linear_policy.cpp` 和 View adapter，观察 backend 选择与跨章数据交接。Q4_K、Q6_K、Q8_0 的 block 尺寸不同，不能共享同一套位解码代码。

## 运行与验证

从仓库根目录执行：

```bash
bash code/chapter05_quantization/scripts/run.sh
```

Q4_K/Q6_K 已完成 CPU block reference、scalar fused、AVX2/FMA fused backend，以及 GPU dequantize + SGEMM correctness 基线。Q8_0 已完成 CPU block reference、scalar fused、统一 policy 和真实 SmolVLM 32 层 Decoder/LM Head 对照；AVX2 与 GPU backend 尚未实现。`QuantizedLinearPolicy` 管理 CPU backend，GPU 路径由 `CudaExecutionContext` 编排。

## 扩展入口

新增量化格式时先在 Core 登记 block 布局和字节数校验，再提供独立反量化 reference，最后实现 fused 或 SIMD backend。只有 reference 数值测试通过后才能加入 policy；性能结论由 Chapter09 登记，本章 benchmark 只生成原始测量。
