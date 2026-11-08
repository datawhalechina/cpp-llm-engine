# 教程定位与职责边界

本文是教程定位、能力范围和章节结构的维护基准。README 面向学习者介绍项目，章节正文解释技术，模型与外部框架坐标由 Reference 维护。

## 定位

`cpp-llm-engine` 是一套中文优先、以 C++ 为实现主线、CPU 可完整学习、GPU 可选加速的工程教程。目标是构建可 DIY 的单机推理服务框架：组件职责、替换接口和验证方法对学习者可见，可以扩展 Loader、Backend、KV Cache、Scheduler 和 Transport。

可运行承诺限于单机 CPU 主线和可选单卡 CUDA。多 GPU、分布式执行和生产级多租户只作为工程参照，不写成当前能力。Python 仅在能降低算法理解成本时提供短镜像，不形成第二套 Runtime。

主线使用同一条证据链：真实模型基线 → 算法与数据契约 → 最小 C++ 实现 → 测试或性能证据 → 成熟框架坐标。完成主线后，学习者应能连接权重、Tensor、Operator、Decoder、Generation Runtime 和 ServiceCore，并判断实现或优化是否有独立证据。

## 能力范围

| 深度 | 纳入内容 | 验收要求 |
| --- | --- | --- |
| 主线实现 | TinyLlama GGUF、Tokenizer、CPU Decoder、Runtime、量化、性能证据和 HTTP/SSE | CPU 可运行；默认回归不依赖模型、网络或 GPU |
| 扩展实现 | 单卡 NVIDIA GPU backend、SmolVLM 多模态 CPU 路径 | 先有 CPU golden，再验证设备或多模态增量 |
| 教学设计 | Paged KV、Radix Cache、LPM、Overlap Scheduling、结构化输出 | 给出数据结构、状态变化和最小实验 |
| 工程参照 | 多进程、模型并行、Prefill/Decode 分离、Expert Parallel、集群 KV | 说明生产问题、约束和源码坐标 |
| 非目标 | 训练、反向传播、全部模型格式、生产集群和多租户运维 | 不进入主线实现承诺 |

## 模块职责

章节按依赖关系和可验证产物排列。CUDA 和多模态复用主线契约，不建立第二套 Runtime 和 Serving。

| 模块 | 负责的产物 |
| --- | --- |
| Foundation | C++、CMake、测试、CUDA、内存和数据结构前置能力 |
| Chapter00 | 使用成熟实现建立真实模型推理基线 |
| Chapter01–04 | 输入、Tensor、Operator、Attention、FFN 和 Decoder |
| Chapter05 | 量化表示、解码、Quantized Linear 和 backend 选择 |
| Chapter06 | Request、Prefill/Decode、Token History、采样和单请求 Runtime |
| Chapter07 | Block Pool、BlockTable、分页 K/V 和缓存事务 |
| Chapter08 | 请求队列、优先级、Continuous Batching、Overlap 和 Tensor batch |
| Chapter09 | Correctness、benchmark、profiling、瓶颈归因和 A/B 决策 |
| Chapter10 | ServiceCore、worker、HTTP/SSE、metrics 和资源生命周期 |
| `code/core` | 跨章节、设备无关的稳定契约 |
| `code/engine` | 真实模型与章节资产集成 |
| `code/cuda` | 单卡 CUDA backend 和设备资源契约 |
| `code/multimodal` | 图片输入、Vision Encoder、Projector 和 embedding 拼接 |
| Guides / Reference | 可变学习计划；统一术语、公式、资产和框架坐标 |
| Maintenance | 定位、模板、教学资产和代码审核规则 |

## 结构调整条件

新增内容先判断属于主线实现、扩展实现、教学设计还是工程参照。只有出现新的稳定依赖阶段和独立验收产物时才新增章节。

| 情况 | 默认处理 |
| --- | --- |
| 新算法可由现有契约表达 | 增加正文、测试或 Reference 对照 |
| 新 backend 复用现有语义 | 放入对应 backend，不新增主线章节 |
| 新输入模态复用 LLM Runtime | 扩展输入和模型执行，不复制 Runtime |
| 新框架提供更清楚的实现 | 更新生态映射，不复制其目录结构 |
| 内容同时改变依赖阶段和验收产物 | 评估新增章节或拆章 |

结构调整后必须同步核对章节导读、代码 README、测试 owner、导航和资产索引。外部项目的具体作用和源码位置统一维护在 [`reference/ecosystem`](../reference/ecosystem/README.md)。
