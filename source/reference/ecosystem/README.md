# 推理生态项目索引

先在[训练框架、推理引擎与推理服务](../framework_comparison.md)确定问题所在的系统位置，再按问题选择项目卡，查看源码对象和对应教程章节。

## 按问题选择项目

同一个项目可能跨越多层，但每张卡只保留它对本教程最有价值的阅读角度。模型家族和交付工具单独列出，避免与推理引擎混为一类。

| 当前问题 | 优先阅读 | 补充阅读 |
| --- | --- | --- |
| 本地真实模型与 GGUF | [llama.cpp](./llama_cpp.md) | [Gemma](./gemma.md) |
| C++/CUDA 纵向实现 | [Tiny-VLLM](./tiny_vllm.md) | [TensorRT-LLM](./tensorrt_llm.md) |
| 可读的模型与 Runtime 算法 | [nano-vLLM](./nano_vllm.md) | [vLLM](./vllm.md) |
| Continuous Batching 与 Block KV | [vLLM](./vllm.md) | [Mini-SGLang](./mini_sglang.md) |
| Radix Cache、LPM 与前缀复用 | [Mini-SGLang](./mini_sglang.md) | [SGLang](./sglang.md) |
| GPU kernel、量化和 engine build | [TensorRT-LLM](./tensorrt_llm.md) | [Tiny-VLLM](./tiny_vllm.md) |
| 大规模阶段分离与联合优化 | [DeepInfer](./deepinfer.md) | [SGLang](./sglang.md) |
| 本地产品交付 | [Ollama](./ollama.md) | [llama.cpp](./llama_cpp.md) |

## 完成主线后怎样继续

先回到对应章节确认教学实现和测试，再进入项目卡寻找生产扩展；不要从大型仓库目录反推教程概念。

| 学习目标 | 先复查 | 再阅读 |
| --- | --- | --- |
| 深入 Tensor、算子和 Decoder | Chapter02–05 | Tiny-VLLM、nano-vLLM、TensorRT-LLM |
| 深入缓存、调度与服务 | Chapter06–10 | vLLM、Mini-SGLang、SGLang |
| 继续做性能优化 | Chapter09 的 workload、Profiling 和 A/B | TensorRT-LLM、DeepInfer |
| 扩展模型格式或本地交付 | Chapter00–01、模型支持表 | llama.cpp、Gemma、Ollama |
| 扩展多模态输入 | [多模态学习指南](../../guides/multimodal_inference.md) | llama.cpp、DeepInfer |

## 项目类型

分类只用于避免误读，不代表项目只能承担一种职责。

| 类型 | 项目 | 阅读目的 |
| --- | --- | --- |
| 本地 C++ baseline | llama.cpp | 建立真实推理、GGUF 和 CPU 路径坐标 |
| 教学型实现 | Tiny-VLLM、nano-vLLM、Mini-SGLang | 观察被压缩后的 kernel、模型或 Runtime 骨架 |
| 生产推理框架 | vLLM、SGLang、TensorRT-LLM、DeepInfer | 观察规模扩大后的执行、调度、缓存和部署约束 |
| 模型家族 | Gemma | 观察模型结构怎样影响 loader、shape 和量化 |
| 交付工具 | Ollama | 观察推理能力怎样包装为本地产品 |

## 项目卡包含什么

每张项目卡给出推荐阅读时机、关键源码对象、对应教程章节和官方入口。稳定概念用于建立长期坐标，具体源码以链接的项目版本为准。
