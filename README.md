# 从零构建 C++ LLM 推理引擎

`cpp-llm-engine`

从 GGUF、张量与解码器，到分页 K/V 缓存、调度与推理服务

_A Practical Tutorial for Building a C++ LLM Inference Engine_

[![CPU and docs CI](https://github.com/datawhalechina/cpp-llm-engine/actions/workflows/ci.yml/badge.svg)](https://github.com/datawhalechina/cpp-llm-engine/actions/workflows/ci.yml)

这是一套中文优先、以 C++ 为实现主线的推理引擎工程教程。教程以 TinyLlama 作为默认语言模型基线，连接模型加载、张量、算子、解码器、生成运行时、缓存、调度和服务；每章通过测试、参考结果或性能实验验证实现，并与 llama.cpp、vLLM、SGLang 和 TensorRT-LLM 的工程策略对照。CPU 建立完整推理服务主线，单卡 CUDA 在保持上层语义不变的前提下替换执行后端，SmolVLM 多模态在解码器前扩展视觉输入并复用后续运行时，最终目标是构建一个可 DIY 的单机推理服务框架。

![cpp-llm-engine 教程主图](source/assets/diagrams/tutorial-concept-map.svg)

## 快速开始

第一次学习建议完成三个动作：在 Chapter00 看见一条完整的模型推理路径，在 Chapter01 读取模型配置和权重，再沿 [USAGE.md](USAGE.md) 进入 Tensor、算子、Decoder、Runtime 和服务章节。

先准备 CMake 3.20+、支持 C++17 的编译器和 Git。Linux 可以直接使用下面的命令；Windows 建议进入 WSL2 Ubuntu 后执行。安装方法和常见问题见 [Chapter00 环境准备](source/chapter00_quickstart/01_environment_setup.md)。

```bash
cmake --version
c++ --version
git --version
```

[Foundation](source/foundation/intro.md) 提供 C++ 所有权、CMake 和 CTest 练习。需要补充其中某项基础时先完成对应练习，工具链基础具备后从 Chapter00 开始主线。

先运行使用仓库内置示例的 Chapter00 环境实验。看到 CTest 显示测试通过，说明 C++ 编译和运行环境已经就绪：

```bash
bash code/chapter00_quickstart/run.sh
```

Chapter00 将工具链验证和真实模型推理分成两个步骤。按照 [Chapter00 导读](source/chapter00_quickstart/intro.md) 准备 `llama.cpp`、TinyLlama GGUF 和模型路径后，继续运行真实推理；CPU 环境可以完成教程主线，CUDA 环境用于后续可选加速实验。

准备好 `llama.cpp` 和 TinyLlama GGUF 后，再运行一次真实模型推理，观察 prompt 怎样变成连续输出：

```bash
bash code/chapter00_quickstart/run_real_inference.sh
```

随后进入 Chapter01，开始读取 GGUF 元数据、Tokenizer 配置和模型权重：

```bash
bash code/chapter01_tensor_loader/run.sh
```

完成 Chapter01 后，按照 [USAGE.md](USAGE.md) 的阅读顺序和运行命令继续 Chapter02–10。

## 教程框架

主图展示章节主线和可选扩展，下面的阶段表用于定位每组章节的学习产物。具体页面和运行命令统一由 [USAGE.md](USAGE.md) 导航。

| 阶段 | 章节 | 产物 |
| --- | --- | --- |
| 真实基线 | Foundation（按需）、Chapter00 | TinyLlama 本地推理与工具链基线 |
| 引擎对象 | Chapter01–03 | 模型、张量/视图、算子和执行后端 |
| 模型与生成 | Chapter04–06 | 解码器、量化线性层、请求、预填充/解码和流式输出 |
| 缓存与调度 | Chapter07–08 | 分页 K/V、块表、连续批处理和张量批次 |
| 证据与服务 | Chapter09–10 | 基准测试、性能剖析、服务核心和 HTTP/SSE |
| 可选扩展 | CUDA、多模态 | GPU 执行后端或视觉输入接入现有运行时 |

## 建设进度

更新时间：**2026-11-09**。下面按最新进展在前排列；详细实验条件与复现入口见[实验资产表](source/reference/experiment_assets.md)。

| 时间 | 里程碑 | 状态 | 可验证产物 |
| --- | --- | --- | --- |
| 2026-11-09 | Qwen3 CPU 推理服务闭环 | 完成 | Qwen3-0.6B Loader、Tokenizer、28 层 Decoder、Generation、分页 K/V、Radix 命中与 HTTP/SSE 验收 |
| 2026-11-09 | Adaptive GEMV 适用范围矩阵 | 完成 | EXP-GPU-16 覆盖 `batch=1/2`、上下文 `16/128` 和 Decode `1/8` 步；Tensor Batch 延迟降低约 `12.9%`～`23.1%` |
| 2026-11-09 | CUDA 量化性能证据 | 完成 | Adaptive GEMV 默认路径、自动 `KEEP/REJECT/INCONCLUSIVE` 决策、NCU 原因指标与 EXP-GPU-14/15 |
| 2026-11-09 | 分页缓存与前缀复用 | 完成 | CUDA 分页 Prefill、分页 Decode batch、Radix Cache、共享块引用和缓存感知调度 |
| 2026-11-09 | 学习入口与参考资产整合 | 完成 | CPU、CPU+GPU、多模态路线，术语/公式表，框架对照卡和实验资产表 |
| 2026-11-08 | TinyLlama CUDA 纵向闭环 | 完成 | 22 层 Model/Context、真实生成、分页 GQA、Chapter06 Runtime 与 Chapter10 服务 |
| 2026-11 | 工程验证体系 | 完成 | 默认回归、Release 发布验收、测试资产审计、文档检查和 GitHub Actions CI |
| 2026-10 | 多模态输入闭环 | 完成 | SmolVLM Vision、Projector、Decoder、ServiceCore 和 profiling golden |
| 2026-10 | CPU 主线闭环 | 完成 | TinyLlama 22 层 Decoder、Runtime、HTTP/SSE 和真实模型验收 |
| 2026-08 | 教程骨架与 Foundation | 完成 | 章节主线、基础题库和 benchmark trace |

## 查阅与贡献

学习路线、章节命令和验证层级见 [USAGE.md](USAGE.md)；术语、公式、实验和框架源码坐标见 [Reference](source/reference/README.md)；修改、测试和提交方法见 [CONTRIBUTING.md](CONTRIBUTING.md)。GitHub Actions 持续检查文档、默认回归和发布验收规则。

测试从对象规则逐步扩展到组件协作、跨章主线和真实模型；性能实验建立在正确性验证通过之后。

![教程验证证据图](source/assets/diagrams/test-pipeline.svg)
