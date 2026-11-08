# 05.7 量化实现与成熟框架对照

本章把量化拆成格式解析、block 解码、Linear 执行、backend policy 和数值 A/B。成熟框架还要根据设备能力、模型层和吞吐目标选择量化路径。

## 对象映射

| 教程对象 | 成熟框架中的对应物 | 阅读重点 |
| --- | --- | --- |
| Q4_K/Q6_K/Q8_0 block reader | ggml quantized type/format | scale、min、quant bits 怎样布局 |
| reference dequantize | reference quant kernels | fused 实现用什么结果校验 |
| scalar/AVX2 fused Linear | CPU quantized GEMV/GEMM | 是否避免完整反量化中间量 |
| `QuantizedLinearPolicy` | quantization method/backend dispatch | dtype、硬件和 shape 怎样选择实现 |
| Decoder/ServiceCore A/B | engine accuracy/performance regression | 优化是否同时守住输出与性能 |

## 教学实现与生产实现

| 本章可验证对象 | 成熟系统继续增加 | 增加它解决什么问题 |
| --- | --- | --- |
| INT8 教学格式、真实 Q4_K/Q6_K/Q8_0 | AWQ、GPTQ、FP8 等多格式 | 适配不同模型资产和硬件能力 |
| block reference 与 fused CPU Linear | GPU quantized GEMM | 降低设备端带宽和计算成本 |
| Float32 激活输入 | SmoothQuant、W8A8、FP8 activation | 让低精度表示继续进入矩阵计算 |
| scalar/AVX2 显式策略 | 自动硬件调优和 tactic selection | 为不同 shape 选择稳定高效实现 |
| Tensor、Decoder、服务级回归 | 校准和训练后量化工具链 | 在部署前控制精度损失 |

## 继续阅读源码

1. 量化格式的 block layout 在 loader、Tensor 和 kernel 之间怎样保持一致？
2. fused Linear 省掉了哪个中间量，误差基线来自哪里？
3. backend policy 的选择依据是硬件能力、shape、格式，还是三者共同决定？

| 项目卡 | 在本章中的用途 |
| --- | --- |
| [llama.cpp](../reference/ecosystem/local_and_delivery.md#llama-cpp) | Q4_K/Q6_K/Q8_0 block 与 CPU 量化 kernel |
| [vLLM](../reference/ecosystem/production_frameworks.md#vllm) | quant method 与模型 Linear 的绑定 |
| [TensorRT-LLM](../reference/ecosystem/production_frameworks.md#tensorrt-llm) | GPU 量化 kernel、workspace 和执行策略 |

## 验证与衔接

```bash
bash code/chapter05_quantization/scripts/run.sh
```

先用 component 测试确认三种格式共享 Linear 契约，再用 integration 测试确认 Chapter02 激活 View 不依赖具体量化格式。二者分别对应成熟框架中的格式实现和模型执行接口；[`05.8 量化在生成运行时中的位置`](./08_quantization_in_serving_runtime.md) 将继续追踪这项接口怎样被 Decoder 和 Generation Runtime 调用。
