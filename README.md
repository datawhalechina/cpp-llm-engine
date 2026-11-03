# cpp-llm-engine

用 C++ 构建 LLM 推理引擎

_A Practical Tutorial for Building a C++ LLM Inference Engine_

[![CPU and docs CI](https://github.com/lynnyulinlin-debug/cpp-llm-engine/actions/workflows/ci.yml/badge.svg)](https://github.com/lynnyulinlin-debug/cpp-llm-engine/actions/workflows/ci.yml)

这是一套中文优先、以 C++ 为实现主线、CPU 可完整学习、GPU 可选加速的推理引擎工程教程。目标是从真实模型基线出发，逐步构建可以自行修改和验证的推理服务框架。

主图从上到下展示学习阶段、代码骨架和能力扩展。CPU 主线负责形成完整推理服务，CUDA 替换执行后端，多模态扩展输入路径，成熟框架用于核对工程策略。

![cpp-llm-engine 教程主图](source/assets/diagrams/tutorial-concept-map.svg)

## Quick Start

准备好 CMake、C++17 编译器、Node.js 和 npm 后，先运行无需模型文件的轻量回归：

```bash
npm install
npm run test:regression
bash code/chapter00_quickstart/run.sh
```

随后按 [Chapter 00](source/chapter00_quickstart/intro.md) 准备 `llama.cpp` 和 TinyLlama GGUF，运行第一次真实推理。完整学习顺序与可选扩展见 [USAGE.md](USAGE.md)。

## 教程定位

教程沿“真实基线 → 数据与算法契约 → C++ 实现 → 验证证据 → 成熟框架对照”推进。Python、NumPy 和外部项目提供解释或独立参考；可运行范围聚焦单机 CPU、可选单消费级 NVIDIA GPU 和多模态输入扩展，多 GPU、分布式和生产级多租户作为生态参照。

| 教程特点 | 学习者获得什么 | 如何验证 |
| --- | --- | --- |
| 真实模型纵向闭环 | 从 GGUF、Tokenizer、Embedding、Decoder 到 Runtime 和 ServiceCore | smoke、真实模型和服务路径分别验证 |
| C++ 主线、CPU 可完整学习 | 无 GPU 也能理解并运行完整引擎主线，CUDA 和多模态复用上层契约 | 可选能力使用独立构建和验证入口 |
| Correctness-first | 每项实现附带单测、golden、benchmark 或 profiling 证据 | 正确性通过后再比较性能 |
| **章节级框架对照卡** | 把本章对象映射到 `llama.cpp`、vLLM、SGLang、TensorRT-LLM 等真实工程 | 分别标明概念、接口和实现证据 |
| 教学与生产分层 | 识别“已实现”“教学设计”“生产级参照”三种深度 | 生产组件提供约束、项目卡和源码坐标 |
| 代表模型与独立参考 | TinyLlama 用于语言主线，SmolVLM 用于多模态；`llama.cpp` 和 NumPy 提供外部对照 | token、logits、checksum、HTTP/SSE 和 profiling golden |

## 教程结构

章节按以下阶段推进，具体页面和运行命令统一由 `USAGE.md` 导航：

```text
Foundation -> Chapter00 -> Chapter01 -> Chapter02 -> Chapter03 -> Chapter04
    -> Chapter05 -> Chapter06 -> Chapter07 -> Chapter08 -> Chapter09
```

Chapter09 位于主线末尾，也可以在完成 Chapter06 后提前进入服务专题。

| 阶段 | 章节 | 产物 |
| --- | --- | --- |
| 真实基线 | Foundation（按需）、Chapter00 | TinyLlama 本地推理与工具链基线 |
| 引擎对象 | Chapter01-03 | Model、Tensor/View、Operator 和 Backend |
| 模型与 Runtime | Chapter04-06 | Decoder、Request、KV Cache、Scheduler、Streaming 和量化 Linear |
| 证据与服务 | Chapter07-09 | Benchmark、Profiling、ServiceCore 和 HTTP/SSE |
| 可选扩展 | CUDA、多模态 | GPU Backend 或视觉输入接入现有 Runtime |

章节是从输入到服务的纵向学习顺序；数据、算子和系统是贯穿各章的横向观察线。数据层追踪模型与请求状态怎样表示，算子层追踪数值计算怎样执行，系统层追踪多请求怎样调度、分配资源并对外服务。

| 横向视角 | 主要章节 | 重点对象 | 完成后能回答什么 |
| --- | --- | --- | --- |
| 数据层 | Chapter01-02、04、06 | GGUF、Tensor/View、Q/K/V、Context、BlockTable | 数据是什么形状、由谁持有、怎样随请求增长 |
| 算子层 | Chapter03-05、CUDA | MatMul、Attention、FFN、量化 Linear、Backend | 一次计算做什么、由哪个 CPU/GPU 实现执行 |
| 系统层 | Chapter00、06-09 | Runtime、Scheduler、Benchmark、Profiling、ServiceCore | 请求怎样持续推进、测量、回收并通过 HTTP/SSE 输出 |

## 建设进度

更新时间：**2026-11-03**。详细实验条件与复现入口见[实验资产表](source/reference/experiment_assets.md)。

| 时间 | 里程碑 | 状态 | 可验证产物 |
| --- | --- | --- | --- |
| 2026-08 | 教程骨架与 Foundation | 完成 | 章节主线、基础题库、benchmark trace |
| 2026-10 | CPU 主线闭环 | 完成 | TinyLlama 22 层 Decoder、Runtime、HTTP/SSE 和真实模型验收 |
| 2026-10 | 多模态输入闭环 | 完成 | SmolVLM Vision、Projector、Decoder、ServiceCore 和 profiling golden |
| 2026-11 | CUDA 后端 | 建设中 | CUDA 资源、量化 Linear、RoPE、KV、GQA 和合成 Decoder；下一步接真实模型 |
| 2026-11 | 工程验证体系 | 完成 | 默认回归、Release 契约、测试资产审计、文档检查和 GitHub Actions CI |

## 验证、查阅与贡献

默认 CPU 回归不依赖 GGUF、网络下载或 GPU；真实模型、transport、性能和 CUDA 使用各自入口补充证据。GitHub Actions 会在 `main` push 和 pull request 上执行文档检查与默认回归。

测试先检查单个对象的契约和不变量，再检查组件协作与跨章主线；真实模型和性能实验建立在正确性通过之后。图中的每一层对应不同问题，不能用一次端到端成功替代底层契约测试。

术语、公式、实验和框架坐标集中在 [Reference](source/reference/README.md)；修改与提交规则见 [CONTRIBUTING.md](CONTRIBUTING.md)。

```bash
npm install
npm run docs:check-style
npm run docs:check-links
npm run test:regression
npm run test:release-contracts
```

![教程验证证据图](source/assets/diagrams/test-pipeline.svg)
