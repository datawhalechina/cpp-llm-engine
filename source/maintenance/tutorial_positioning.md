# 教程定位与职责边界

本文是教程定位、范围和结构调整的维护基准。README 面向学习者说明可以获得什么；章节正文解释稳定知识；建设优先级和范围变更以本文为准。

## 定位

`cpp-llm-engine` 是一套中文优先、以 C++ 为实现主线、CPU 可完整学习、GPU 可选加速的工程教程。教程最终目标是构建可 DIY 的单机推理服务框架：学习者可以替换或扩展 Loader、Backend、KV Cache、Scheduler 和 Transport，并用共通契约、测试和性能证据判断改动是否成立。

“可 DIY”表示组件职责、替换接口和验证方法对学习者可见，不表示当前已具备生产集群能力。可运行承诺仍限于单机 CPU 主线和可选单卡 CUDA；多 GPU、分布式执行和生产级多租户作为工程参照。

教程使用同一条证据链组织内容：先用真实模型建立外部基线，再拆解算法与数据契约，完成最小 C++ 实现，通过测试或 benchmark 验证，最后映射到成熟框架的工程设计。Python 只在能显著降低算法理解成本时提供短小镜像，不形成第二条实现主线。

学习者完成主线后应能：

- 从 GGUF、Tokenizer 和 Tensor 开始解释模型如何进入执行路径
- 用 C++ 连接 Operator、Decoder、Generation Runtime 和 ServiceCore
- 区分模型计算、请求状态、KV 管理、调度、量化和 transport
- 用 golden、单元测试、benchmark 和 profiling 判断实现是否正确、优化是否成立
- 在 `llama.cpp`、vLLM、SGLang、TensorRT-LLM 等项目中找到对应工程对象

## 能力范围

可执行主线必须提供真实输入、C++ 产物和验证证据；大规模生产能力只解释问题、设计和源码坐标，不能写成仓库已经具备的功能。

| 深度 | 纳入内容 | 验收要求 |
| --- | --- | --- |
| 主线实现 | TinyLlama GGUF、真实 Tokenizer、CPU Decoder、Generation Runtime、量化、benchmark、profiling、HTTP/SSE | CPU 环境可运行；默认回归不依赖模型、网络或 GPU |
| 扩展实现 | 单消费级 NVIDIA GPU backend、SmolVLM 多模态 CPU 路径 | 先有 CPU golden，再验证设备或多模态增量 |
| 教学设计 | Paged KV、Radix Cache、LPM、Overlap Scheduling、结构化输出 | 给出数据结构、状态变化、伪代码或最小实验 |
| 工程参照 | 多进程 worker、模型并行、Prefill/Decode 分离、Expert Parallel、集群 KV 存储 | 说明生产问题、约束和成熟框架源码位置 |
| 非目标 | 训练、反向传播、支持全部模型格式、生产级多租户和集群运维 | 不进入主线实现承诺 |

## 模块职责

章节顺序按依赖关系和可验证产物固定，不按外部项目目录或技术热点重新编号。CUDA 和多模态是复用主线契约的扩展方向，不建立第二套 Runtime 和 Serving。

| 模块 | 职责 | 不承担的职责 |
| --- | --- | --- |
| Foundation | C++、CMake、测试、CUDA、内存和数据结构前置能力 | 完整模型和 Runtime |
| Chapter00 | 使用成熟实现建立真实模型推理基线 | 自研引擎实现 |
| Chapter01-04 | 输入、Tensor、Operator、Attention、FFN 和 Decoder 执行核心 | 请求调度和 HTTP 服务 |
| Chapter05 | 量化表示、解码、Quantized Linear 和 backend 选择 | 通用性能结论 |
| Chapter06 | Request、Prefill/Decode、Token History、采样和单请求 Generation Runtime | 分页 KV 资源分配和多请求调度 |
| Chapter07 | Block Pool、BlockTable、分页 K/V payload 和缓存事务 | 请求选择和服务协议 |
| Chapter08 | 请求队列、优先级、Continuous Batching、Overlap 和 Tensor batch | 性能结论和 HTTP transport |
| Chapter09 | Correctness、benchmark、profiling、瓶颈归因和 A/B 决策 | 无证据的优化建议 |
| Chapter10 | ServiceCore、worker、HTTP/SSE、metrics 和资源生命周期 | 分布式服务平台 |
| `code/core` | 跨章节、设备无关的稳定契约 | 具体 kernel |
| `code/engine` | 真实模型和章节资产集成 | 复制章节教学实现 |
| `code/cuda` | 单卡 CUDA backend 和设备资源契约 | 多 GPU、NCCL 和集群调度 |
| `code/multimodal` | 图片输入、Vision Encoder、Projector 和 embedding 拼接 | 第二套 LLM Runtime |
| Guides | 环境操作、学习计划和复现流程 | 核心算法正文 |
| Reference | 术语、公式、资产表和框架映射 | 第二套章节主线 |
| Maintenance | 状态、路线、模板和审核结论 | 学习者教程内容 |

## 教学证据链

正文按学习对象选择表达方式，不机械要求每章都有 Python。Loader、Storage、Benchmark 和 Profiling 更适合直接使用文件布局、对象关系或测量表；Attention、Scheduler、量化解码和缓存算法可以先给短算法镜像。

| 环节 | 必需性 | 内容要求 |
| --- | --- | --- |
| 问题与产物 | 必需 | 输入、输出、最小命令和成功标准 |
| 算法镜像 | 按需 | 自维护的短 Python 伪代码、状态图或步骤表 |
| C++ 实现 | 必需 | 对象、ownership、状态、backend 和错误契约 |
| 验证证据 | 必需 | 单测、golden、真实 baseline、benchmark 或 profiling |
| 框架坐标 | 必需 | intro 保留小卡片，详细映射集中到 Reference |

## 外部项目职责

外部项目按其最适合解释的问题使用，不把学习者带入多套并行源码主线。框架版本变化只更新 Reference 的源码坐标，不反向改变本教程的稳定技术契约。

| 项目 | 给教程提供的主要输入 | 主要落点 |
| --- | --- | --- |
| `llama.cpp` | 本地真实推理、GGUF、CPU 与多模态 baseline | Chapter00-01、06、多模态 |
| Tiny-VLLM | C++/CUDA 纵向实现和零基础 kernel 拆解 | Foundation、Chapter03-06、CUDA |
| nano-vLLM | 可读的 Python 模型与 Runtime 算法骨架 | Chapter04-05、08 |
| vLLM | EngineCore、continuous batching、Block KV 和生产执行器 | Chapter06、07-09 |
| Mini-SGLang | 可读的 Prefill/Decode、Radix Cache 和 overlap scheduling | Chapter06、09 |
| SGLang | LPM、RadixAttention、HiCache 和生产调度扩展 | Chapter06、09、Reference |
| TensorRT-LLM | GPU kernel、量化、engine build 和部署边界 | Chapter03-04、06、08 |
| DeepInfer | 模型、kernel、通信、调度和集群拓扑的联合设计 | Chapter09、多模态完成后的进阶案例 |

## 结构变更原则

新增内容先判断属于主线实现、扩展实现、教学设计还是工程参照。只有出现新的稳定依赖阶段和独立可验证产物时才考虑新增章节；成熟框架增加功能、出现新的项目或源码目录变化，不构成章节重排理由。

| 变更 | 默认决策 |
| --- | --- |
| 新算法可由现有契约表达 | 增加正文小节、测试或 Reference 对照 |
| 新 backend 复用现有语义 | 放入 `code/cuda` 或对应 backend，不新增主线章节 |
| 新输入模态复用 LLM Runtime | 放入 `code/multimodal`，只增加输入和模型执行扩展 |
| 新框架提供更清楚的实现 | 更新框架映射，不复制其目录结构 |
| 内容同时改变依赖阶段和验收产物 | 评估新增章节或拆章 |
