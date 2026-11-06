# 05.0 用仅权重量化减少权重存储并建立计算路径

## 本章导读

Chapter 04 已经固定 Decoder、Attention 和 FFN 的计算结构；本章在 Linear 内部用低比特格式保存真实权重，并比较浮点路径与量化路径的输出误差、存储大小和执行时间。Chapter 06 会在不改变请求语义的前提下持续调用这条模型执行路径。

| 本章实现 | 成熟系统继续增加 | 深入入口 |
| --- | --- | --- |
| INT8 教学路径、GGUF Q4_K/Q6_K/Q8_0 解码、CPU reference/fused/scalar batch/SIMD 和 CUDA 非融合基线 | 校准、混合精度、专用 batch SIMD、融合 GPU kernel 和多格式调度 | [`05.7 量化框架对照`](./07_tensorrt_gemma_vllm_comparison.md) |

| 学习线 | 本章怎样学习 | 可验证产物 |
| --- | --- | --- |
| A 教学证据线 | 用小型 INT8、Q4_K/Q6_K/Q8_0 block 和 CPU reference 看清 scale、反量化与误差 | 固定反量化数值、浮点/量化输出误差和失败状态 |
| B 真实执行线 | 读取 GGUF packed 权重，通过 fused/SIMD backend 接入 TinyLlama 22 层 Decoder | 固定 logits、greedy token、Backend policy 和 Decoder 数值对照 |

| 本章新术语 | 先这样理解 | 首次解释 |
| --- | --- | --- |
| 量化（quantization） | 用较少的整数位保存近似权重，减少模型体积和内存带宽 | [`05.1 用整数近似浮点权重`](./01_quantization_basics.md) |
| 缩放因子（`scale`） | 把整数值映射回近似浮点范围的比例 | [`05.1 用整数近似浮点权重`](./01_quantization_basics.md) |
| 反量化（dequantization） | 根据整数值和 scale 恢复近似浮点值 | [`05.4 反量化算子`](./04_dequantize_op.md) |
| 仅权重量化（weight-only） | 权重使用低比特格式，输入与输出仍按浮点计算 | [`05.5 量化线性层`](./05_linear_with_quant.md) |
| 融合路径（fused path） | 读取量化 block 时直接乘加，不生成完整浮点权重副本 | [`05.5 量化线性层`](./05_linear_with_quant.md) |

![Chapter 05 主图：浮点与量化 Linear 路径对照](../assets/diagrams/chapter05-main.svg)

## 开始学习

先运行不依赖真实模型文件的默认入口：

```bash
bash code/chapter05_quantization/scripts/run.sh
```

脚本会先运行 `chapter05_quantization_demo`，输出反量化值 `0.1`、`2x2` Linear 结果、`value(0,0)=0.5` 和 `value(1,1)=2.2`；随后 CTest 比较浮点、reference、fused、公共激活 View 与错误输入路径。真实 TinyLlama 提供 Q4_K/Q6_K 验证，Q8_0 的真实模型证据来自可选的 SmolVLM CPU 路径。

| 顺序 | 路线 | 页面 | 先看源码 | 读完后应看到什么 |
| --- | --- | --- | --- | --- |
| 1 | A/B 公共 | [`05.1 量化基础`](./01_quantization_basics.md)、[`05.2 量化 Tensor`](./02_extend_tensor.md) | <a href="../_code/chapter05_quantization/src/tensor.cpp"><code>tensor.cpp</code></a> | 浮点值怎样变成整数、scale 和量化误差 |
| 2 | A→B | [`05.3 接收量化权重`](./03_load_quantized_weights.md)、[`05.4 反量化算子`](./04_dequantize_op.md) | <a href="../_code/chapter05_quantization/src/q4k_dequantize.cpp"><code>q4k_dequantize.cpp</code></a>、<a href="../_code/chapter05_quantization/src/q8_0_dequantize.cpp"><code>q8_0_dequantize.cpp</code></a> | 不同 packed block 怎样按各自布局恢复可计算数值 |
| 3 | B | [`05.5 量化线性层`](./05_linear_with_quant.md) | <a href="../_code/chapter05_quantization/src/quantized_linear_view_adapter.cpp"><code>quantized_linear_view_adapter.cpp</code></a>、<a href="../_code/chapter05_quantization/src/quantized_linear_policy.cpp"><code>quantized_linear_policy.cpp</code></a> | 公共激活 View 怎样进入 Q4_K/Q6_K/Q8_0 backend policy |
| 4 | A/B 验证 | [`05.6 精度验证`](./06_accuracy_validation.md) | <a href="../_code/chapter05_quantization/tests/chapter05_quantization_test.cpp"><code>chapter05_quantization_test.cpp</code></a> | 同一输入下量化输出与浮点基线的误差是否可接受 |
| 5 | B | [`05.7 框架对照`](./07_tensorrt_gemma_vllm_comparison.md)、[`05.8 Serving Runtime 中的位置`](./08_quantization_in_serving_runtime.md) | <a href="../_code/core/include/cpp_llm_engine/core/quantization_contract.h"><code>quantization_contract.h</code></a> | 量化后端怎样保持上层 Linear 和 Runtime 接口不变 |

## 验证与下一步

本章留下了可与浮点基线对照的量化 Linear 后端；[`Chapter 06`](../chapter06_generation/next/intro.md) 将把真实 Decoder 接入 `prefill -> decode -> output` 生成循环。

| 验收层级 | 执行入口 | 通过条件 | 本章要求 |
| --- | --- | --- | --- |
| L1 默认 CPU | `bash code/chapter05_quantization/scripts/run.sh` | 两个 CTest 覆盖 Q4_K/Q6_K/Q8_0 backend、rank-1/2 激活 View、错误 payload 和失败保持 | 必做 |
| L2 真实模型 | `bash code/chapter05_quantization/scripts/run_real_decoder.sh` | 真实 TinyLlama 量化权重经过 22 层后得到固定 logits 和 greedy token | 有模型时必做 |
| L3 可选 CUDA | `CUDA_HOME=/usr/local/cuda-12.8 bash code/cuda/scripts/run.sh` | 设备端 Q4_K/Q6_K 反量化、Float32 Workspace 和 cuBLAS 结果匹配 CPU reference | 有 NVIDIA GPU 时选做 |
