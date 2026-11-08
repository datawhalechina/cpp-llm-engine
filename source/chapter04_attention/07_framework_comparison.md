# 04.7 解码器成熟框架对照

本章把 Chapter01 的模型材料、Chapter02 的张量视图和 Chapter03 建立的算子分层方法连接到解码器。教学线性层使用本地可读循环，真实量化投影和 GPU 路径再接入独立计算后端。对照成熟项目时，先核对模型层的数据流，再观察模型执行器、注意力后端、请求执行上下文和 GPU 内核如何接入。

## 对象映射

TinyLlama/Llama 决定本章真实模型的数学结构，llama.cpp 提供同模型 CPU 数值基线。nano-vLLM 适合阅读紧凑前向控制流，vLLM 和 TensorRT-LLM 展示生产执行路径；Gemma 用于比较另一种模型家族怎样调整解码器块。

| 教程对象 | 成熟项目中的对应物 | 阅读重点 |
| --- | --- | --- |
| `DecoderLayer` | Llama 系解码器块 | 归一化、注意力、残差、FFN 的顺序和权重布局 |
| Q/K/V 投影 | 模型注意力输入准备 | 注意力头数、GQA 布局、位置和权重形状 |
| `AttentionBackend` | vLLM 注意力后端、TensorRT-LLM 内核 | 掩码、缓存元数据与内核的接口 |
| 连续 `KvCache` | 分页/分块 KV Cache 的数值基础 | 预填充和逐词元解码怎样复用 K/V |
| `DecoderLayerContract` | 模型配置与执行器约束 | 生成运行时如何检查模型结构能力 |
| `RealLlamaCpuModel` / `RealLlamaCpuContext` | 模型执行器与请求级执行状态 | 共享权重和请求级位置/KV 怎样分离 |
| 最终 RMSNorm / LM Head | 最终归一化与词表投影 | 隐藏状态怎样变成最终候选分数 |

## 教学实现与生产实现

教学实现固定单层数据流和 CPU 数值基线；成熟系统在相同语义上增加模型注册、分页缓存、融合内核和并行执行。

| 本章可验证对象 | 成熟系统继续增加 | 后续入口 |
| --- | --- | --- |
| 单层前置归一化与 22 个 TinyLlama 解码器块 | 多模型架构注册和并行执行计划 | [`运行真实 TinyLlama 解码器`](./06_real_tinyllama_decoder.md) |
| CPU 因果注意力与稳定 softmax | FlashAttention、融合注意力、GPU 工作区 | CUDA 计算后端 |
| 位置偏移与连续 K/V | 分页 K/V、前缀复用和缓存感知调度 | Chapter07、Chapter08 |
| 单层结构和数值测试 | 张量并行、流水线并行、跨设备通信 | 分布式扩展 |

## 继续阅读源码

阅读项目卡中的源码入口时，沿“模型层 → 注意力 → 计算后端”追踪一次前向计算：

1. 模型层为 Q/K/V、位置和残差准备了哪些输入？
2. 注意力后端接收数值张量，还是同时接收缓存与调度元数据？
3. 缓存形状由模型结构决定，物理分配又由哪个运行时对象决定？
4. 预填充和逐词元解码复用同一层权重时，输入形状与元数据有何差异？

| 项目卡 | 在本章中的用途 |
| --- | --- |
| [llama.cpp](../reference/ecosystem/llama_cpp.md) | 核对 TinyLlama/Llama 的 CPU 模型执行和数值结果 |
| [nano-vLLM](../reference/ecosystem/nano_vllm.md) | 先阅读紧凑的模型与前向控制流 |
| [vLLM](../reference/ecosystem/vllm.md) | 观察模型层、注意力后端和缓存元数据 |
| [TensorRT-LLM](../reference/ecosystem/tensorrt_llm.md) | 观察 GPU 运行时、注意力和融合内核 |
| [Gemma](../reference/ecosystem/gemma.md) | 比较模型家族变化怎样影响归一化、位置编码和 FFN |
