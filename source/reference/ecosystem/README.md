# 推理生态项目索引

先在[推理引擎系统与执行架构](../architecture.md)确定问题所在的系统位置，再按问题选择项目卡，查看源码对象和对应教程章节。

## 按问题选择项目

同一个项目可能跨越多层，但每张卡只保留它对本教程最有价值的阅读角度。模型家族统一放在[模型支持表](../model_support.md)，不与推理引擎项目卡混为一类。

| 当前问题 | 优先阅读 | 补充阅读 |
| --- | --- | --- |
| 本地真实模型与 GGUF | [llama.cpp](./local_and_delivery.md#llama-cpp) | [模型支持表](../model_support.md) |
| C++/CUDA 纵向实现 | [Tiny-VLLM](./teaching_projects.md#tiny-vllm) | [TensorRT-LLM](./production_frameworks.md#tensorrt-llm) |
| 可读的模型与运行时算法 | [nano-vLLM](./teaching_projects.md#nano-vllm) | [vLLM](./production_frameworks.md#vllm) |
| 连续批处理与分页 K/V | [vLLM](./production_frameworks.md#vllm) | [Mini-SGLang](./teaching_projects.md#mini-sglang) |
| Radix Cache、LPM 与前缀复用 | [Mini-SGLang](./teaching_projects.md#mini-sglang) | [SGLang](./production_frameworks.md#sglang) |
| GPU 计算核、量化和引擎构建 | [TensorRT-LLM](./production_frameworks.md#tensorrt-llm) | [Tiny-VLLM](./teaching_projects.md#tiny-vllm) |
| 大规模阶段分离与联合优化 | [DeepInfer](./production_frameworks.md#deepinfer) | [SGLang](./production_frameworks.md#sglang) |
| 本地产品交付 | [Ollama](./local_and_delivery.md#ollama) | [llama.cpp](./local_and_delivery.md#llama-cpp) |

## 完成主线后怎样继续

先回到对应章节确认教学实现和测试，再进入项目卡寻找生产扩展；不要从大型仓库目录反推教程概念。

| 学习目标 | 先复查 | 再阅读 |
| --- | --- | --- |
| 深入 Tensor、算子和 Decoder | Chapter02–05 | Tiny-VLLM、nano-vLLM、TensorRT-LLM |
| 深入缓存、调度与服务 | Chapter06–10 | vLLM、Mini-SGLang、SGLang |
| 继续做性能优化 | Chapter09 的 workload、Profiling 和 A/B | TensorRT-LLM、DeepInfer |
| 扩展模型格式或本地交付 | Chapter00–01、模型支持表 | llama.cpp、Ollama |
| 扩展多模态输入 | [多模态学习计划](../../guides/multimodal_learning_plan.md) | llama.cpp、DeepInfer |

## 项目类型

分类只用于避免误读，不代表项目只能承担一种职责。

| 类型 | 项目 | 阅读目的 |
| --- | --- | --- |
| 本地 C++ 基线 | llama.cpp | 对照真实推理、GGUF 和 CPU 执行路径 |
| 教学型实现 | Tiny-VLLM、nano-vLLM、Mini-SGLang | 观察被压缩后的 kernel、模型或 Runtime 骨架 |
| 生产推理框架 | vLLM、SGLang、TensorRT-LLM、DeepInfer | 观察规模扩大后的执行、调度、缓存和部署约束 |
| 交付工具 | Ollama | 观察推理能力怎样包装为本地产品 |

## 项目卡包含什么

项目按本地交付、教学实现和生产框架分为三页。每个项目使用“阅读目的、重点对象、对应章节”模板；正文解释稳定概念，具体源码位置以链接的项目版本为准。
