# 05.0 用仅权重量化减少权重存储并建立计算路径

## 本章导读

Chapter 04 已经固定 Decoder、Attention 和 FFN 的计算结构；本章在 Linear 内部用低比特格式保存真实权重，并比较浮点路径与量化路径的输出误差、存储大小和执行时间。Chapter 06 会在不改变请求语义的前提下持续调用这条模型执行路径。

| 本章实现 | 成熟系统继续增加 | 深入入口 |
| --- | --- | --- |
| INT8 教学路径、九种 GGUF 量化格式的 CPU reference/fused、可选 SIMD，以及 Q2_K/Q3_K/Q4_K/Q6_K/IQ4_NL CUDA GEMV | 校准、混合精度、更高吞吐量化 GEMM 和多格式调度 | [`05.7 量化框架对照`](./07_tensorrt_gemma_vllm_comparison.md) |

| 学习线 | 本章怎样学习 | 可验证产物 |
| --- | --- | --- |
| A 教学证据线 | 用小型 INT8 与九种 GGUF block 的 CPU reference 看清 scale、反量化与误差 | 固定反量化数值、浮点/量化输出误差和失败状态 |
| B 真实执行线 | 读取 GGUF packed 权重，将 TinyLlama、Qwen1.5-MoE 和 DeepSeek-V2-Lite 所需格式接入统一 Linear | 与 llama.cpp 对照 logits 数值指标、专家选择和 greedy token 序列 |
| G GPU 执行线 | 保持 packed 权重和 Linear 语义不变，单 token 与小批量路径融合解码、点积和归约 | Q2_K/Q3_K/Q4_K/Q6_K/IQ4_NL 设备端结果；TinyLlama/Qwen3 真实生成与性能证据 |
| M 多模态复用线 | 复用 Q8_0 解码与 Linear 契约处理 SmolVLM 视觉权重 | 视觉 Encoder、Projector 和语言模型权重通过真实模型测试 |

| 本章新术语 | 先这样理解 | 首次解释 |
| --- | --- | --- |
| 量化（quantization） | 用较少的整数位保存近似权重，减少模型体积和内存带宽 | [`05.1 用整数近似浮点权重`](./01_quantization_basics.md) |
| 缩放因子（`scale`） | 把整数值映射回近似浮点范围的比例 | [`05.1 用整数近似浮点权重`](./01_quantization_basics.md) |
| 反量化（dequantization） | 根据整数值和 scale 恢复近似浮点值 | [`05.4 反量化算子`](./04_dequantize_op.md) |
| 仅权重量化（weight-only） | 权重使用低比特格式，输入与输出仍按浮点计算 | [`05.5 量化线性层`](./05_linear_with_quant.md) |
| 混合量化（mixed quantization） | 不同 Tensor 根据精度和执行需求使用不同量化格式 | [`05.1 用整数近似浮点权重`](./01_quantization_basics.md) |
| K-quant | ggml 的分块量化格式族，其中 `K` 与 K/V Cache 无关 | [`05.1 用整数近似浮点权重`](./01_quantization_basics.md) |
| 融合路径（fused path） | 读取量化 block 时直接乘加，不生成完整浮点权重副本 | [`05.5 量化线性层`](./05_linear_with_quant.md) |

![Chapter 05 主图：浮点与量化 Linear 路径对照](../assets/diagrams/chapter05-main.svg)

## 开始学习

先区分权重量化、激活量化和 K/V Cache 量化，再从 INT8 的整数、缩放因子和反量化关系入手，沿着 GGUF block、量化 Linear 和 backend policy 进入真实 Decoder。TinyLlama 提供 Q4_K/Q6_K 证据，SmolVLM 提供 Q8_0 证据，Qwen1.5-MoE 验证 Q3_K/Q4_K/Q5_0/Q5_1/Q5_K 混合量化，DeepSeek-V2-Lite 再引入 Q2_K 与 IQ4_NL。

| 顺序 | 路线 | 页面 | 先看源码 | 读完后应看到什么 |
| --- | --- | --- | --- | --- |
| 1 | A/B 公共 | [`05.1 量化基础`](./01_quantization_basics.md)、[`05.2 量化 Tensor`](./02_extend_tensor.md) | <a href="https://github.com/datawhalechina/cpp-llm-engine/blob/main/code/chapter05_quantization/src/tensor.cpp"><code>tensor.cpp</code></a> | 浮点值怎样变成整数、scale 和量化误差 |
| 2 | A→B | [`05.3 接收量化权重`](./03_load_quantized_weights.md)、[`05.4 反量化算子`](./04_dequantize_op.md) | <a href="https://github.com/datawhalechina/cpp-llm-engine/blob/main/code/chapter05_quantization/src/q2k_dequantize.cpp"><code>q2k_dequantize.cpp</code></a>、<a href="https://github.com/datawhalechina/cpp-llm-engine/blob/main/code/chapter05_quantization/src/iq4nl_dequantize.cpp"><code>iq4nl_dequantize.cpp</code></a> | super-block、普通 block 与非线性码表怎样恢复数值 |
| 3 | B/G/M | [`05.5 量化线性层`](./05_linear_with_quant.md) | <a href="https://github.com/datawhalechina/cpp-llm-engine/blob/main/code/chapter05_quantization/src/quantized_linear_view_adapter.cpp"><code>quantized_linear_view_adapter.cpp</code></a>、<a href="https://github.com/datawhalechina/cpp-llm-engine/blob/main/code/chapter05_quantization/src/quantized_linear_policy.cpp"><code>quantized_linear_policy.cpp</code></a> | 相同 Linear 契约怎样连接 CPU policy、CUDA 执行上下文和多模态 Q8_0 权重 |
| 4 | A/B 验证 | [`05.6 精度验证`](./06_accuracy_validation.md) | <a href="https://github.com/datawhalechina/cpp-llm-engine/blob/main/code/chapter05_quantization/tests/chapter05_quantization_test.cpp"><code>chapter05_quantization_test.cpp</code></a> | 同一输入下量化输出与浮点基线的误差是否可接受 |
| 5 | B | [`05.7 框架对照`](./07_tensorrt_gemma_vllm_comparison.md)、[`05.8 Serving Runtime 中的位置`](./08_quantization_in_serving_runtime.md) | <a href="https://github.com/datawhalechina/cpp-llm-engine/blob/main/code/core/include/cpp_llm_engine/core/quantization_contract.h"><code>quantization_contract.h</code></a> | 量化后端怎样保持上层 Linear 和 Runtime 接口不变 |

## 验证与下一步

本章留下了可与浮点基线对照的量化 Linear 后端；[`Chapter 06`](../chapter06_generation/next/intro.md) 将把真实 Decoder 接入 `prefill -> decode -> output` 生成循环。

| 验收层级 | 执行入口 | 通过条件 | 本章要求 |
| --- | --- | --- | --- |
| L1 默认 CPU | `bash code/chapter05_quantization/scripts/run.sh` | 四个 CTest 覆盖本章 backend、DeepSeek 格式、Chapter02 激活 View 集成和 Core Tensor 契约 | 必做 |
| L2 真实模型 | `bash code/chapter05_quantization/scripts/run_real_decoder.sh` | TinyLlama 22 层输出满足与 llama.cpp 约定的余弦相似度、误差和 greedy 序列条件 | 有模型时必做 |
| L3 可选 CUDA | `CUDA_HOME=/usr/local/cuda-12.8 bash code/cuda/scripts/run.sh` | 单 token 与批量融合 GEMV 均匹配 CPU reference | 有 NVIDIA GPU 时选做 |
