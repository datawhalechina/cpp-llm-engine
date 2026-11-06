# 训练框架、推理引擎与推理服务

训练框架产出模型，推理引擎执行模型，推理服务持续接收和调度请求。本教程位于推理引擎到最小推理服务之间：学习者亲手实现关键 C++ 对象和执行路径，再借助成熟项目理解生产系统增加了什么。

## 三类系统怎样衔接

同一个模型从训练产物进入在线服务，需要经历权重加载、模型执行、生成运行时和服务接入。Python 与 C++ 是常见的职责分工，不等于“Python 是框架、C++ 是引擎”：判断边界时应看对象负责策略编排，还是负责计算、内存和资源生命周期。

| 系统 | 主要输入 | 核心职责 | 典型输出 |
| --- | --- | --- | --- |
| 训练框架 | 数据、模型定义、优化目标 | 自动求导、优化器、训练和 checkpoint | 模型权重与配置 |
| 推理引擎 | 权重、token、运行配置 | forward、KV、采样、调度和设备执行 | token、logits、运行指标 |
| 推理服务 | 请求、模型实例、资源预算 | admission、并发、流式输出、协议和观测 | API/SSE 响应与服务指标 |
| 本教程 | 真实模型基线与最小 C++ 契约 | 连接 loader、operator、decoder、runtime 和 ServiceCore | 可读、可改、可验证的教学实现 |

## 本教程覆盖到哪里

可运行能力和生产系统设计必须分开陈述。CPU 主线需要在普通开发环境中完成闭环；GPU、多模态和复杂调度在已有契约上增量建设；集群能力只提供设计问题和源码坐标。

| 范围 | 本教程的处理方式 | 代表入口 |
| --- | --- | --- |
| 模型输入与执行 | C++ 可运行实现 | Chapter01-04 |
| Generation Runtime | C++ 可运行实现和请求级测试 | Chapter06 |
| 量化与设备 backend | CPU 完整基线、单卡 CUDA 增量 | Chapter05、`code/cuda` |
| K/V 与调度 | CPU 分页数据面、连续批处理和真实 Tensor batch | Chapter07-08 |
| Correctness 与性能 | 可复现 benchmark 和 profiling | Chapter09 |
| 最小推理服务 | ServiceCore、HTTP/SSE、metrics | Chapter10 |
| 多模态 | 复用现有 Runtime 的输入与模型扩展 | `code/multimodal` |
| 分布式与集群服务 | 原理、约束和成熟框架坐标 | Reference |

## 外部项目分别提供什么

这些项目不处于同一抽象层，也不构成从“小项目”到“大项目”的固定学习顺序。阅读时先确定当前问题属于模型执行、Runtime、缓存调度、GPU backend 还是集群部署，再选择一个主参照。

| 项目 | 最适合回答的问题 | 本教程吸收的内容 | 主要对应位置 |
| --- | --- | --- | --- |
| [`llama.cpp`](./ecosystem/llama_cpp.md) | 本地怎样跑通真实模型和 GGUF | 真实 CPU/多模态 baseline、格式与量化坐标 | Chapter00-01、06、多模态 |
| [`Tiny-VLLM`](./ecosystem/tiny_vllm.md) | C++/CUDA 推理路径怎样逐步搭建 | kernel 零基础拆解和纵向实现顺序 | Foundation、Chapter03-06、CUDA |
| [`nano-vLLM`](./ecosystem/nano_vllm.md) | Runtime 算法怎样用短 Python 表达 | 模型执行、分页 KV 和调度算法镜像 | Chapter04、06-08 |
| [`vLLM`](./ecosystem/vllm.md) | 高吞吐引擎怎样组织请求和 KV | EngineCore、continuous batching、Block KV | Chapter06-10 |
| [`Mini-SGLang`](./ecosystem/mini_sglang.md) | 调度和前缀缓存怎样保持可读 | Prefill/Decode、Radix Cache、overlap scheduling | Chapter06-08、10 |
| [`SGLang`](./ecosystem/sglang.md) | 前缀复用怎样进入生产调度 | LPM、RadixAttention、HiCache 和资源预算 | Chapter07-10 |
| [`TensorRT-LLM`](./ecosystem/tensorrt_llm.md) | GPU engine 怎样连接 kernel 与部署 | backend、量化、融合和 engine build | Chapter03-05、09 |
| [`DeepInfer`](./ecosystem/deepinfer.md) | 超大规模系统怎样联合设计计算与拓扑 | 异步调度、并行、阶段分离和开源边界 | Chapter08-10、多模态进阶 |

## 按章节选择对照

每章只选择一个主参照和少量补充参照。框架源码用于检查工程差异，不替代本章 C++ 实现和测试。

| 学习位置 | 主参照 | 重点观察 |
| --- | --- | --- |
| Foundation | Tiny-VLLM | CUDA thread/block、内存和 kernel 启动 |
| Chapter00 | llama.cpp | 模型、Tokenizer、推理命令和真实输出 |
| Chapter01-02 | llama.cpp | GGUF、metadata、Tensor storage/view |
| Chapter03 | Tiny-VLLM、TensorRT-LLM | GEMM 语义、backend 和高性能实现的边界 |
| Chapter04 | nano-vLLM、vLLM | 模型层组装、Attention backend 和权重映射 |
| Chapter05 | llama.cpp、TensorRT-LLM | 量化布局、解码和 fused/non-fused 路径 |
| Chapter06 | nano-vLLM、Mini-SGLang、vLLM | Request、prefill/decode、采样和单请求 Runtime |
| Chapter07 | vLLM、Mini-SGLang、SGLang | Block/Paged K/V、前缀缓存和资源所有权 |
| Chapter08 | vLLM、Mini-SGLang、SGLang | Continuous Batching、Chunked Prefill 和 Overlap |
| Chapter09 | vLLM、TensorRT-LLM、DeepInfer | workload、指标、Profiling 和优化证据 |
| Chapter10 | Mini-SGLang、SGLang、vLLM | admission、worker、网络接口和服务生命周期 |
| 多模态进阶 | llama.cpp、DeepInfer | 本地 Vision 路径和生产阶段分离 |

## 阅读方法

1. 先完成章节的最小命令和验证，不从成熟框架源码开始。
2. 用章节 intro 的对照卡确定需要观察的工程对象。
3. 在[生态项目索引](./ecosystem/README.md)选择项目卡，再进入官方源码。
4. 只迁移能够通过现有 golden、测试或 benchmark 证明价值的设计。
