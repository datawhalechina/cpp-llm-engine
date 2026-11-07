# 从零构建 C++ LLM 推理引擎

`cpp-llm-engine`

从 GGUF、张量与解码器，到分页 K/V 缓存、调度与推理服务

_A Practical Tutorial for Building a C++ LLM Inference Engine_

[![CPU and docs CI](https://github.com/datawhalechina/cpp-llm-engine/actions/workflows/ci.yml/badge.svg)](https://github.com/datawhalechina/cpp-llm-engine/actions/workflows/ci.yml)

这是一套中文优先、以 C++ 为实现主线的推理引擎工程教程。教程以 TinyLlama 作为默认语言模型基线，连接模型加载、张量、算子、解码器、生成运行时、缓存、调度和服务；每章通过测试、参考结果或性能实验验证实现，并与 llama.cpp、vLLM、SGLang 和 TensorRT-LLM 的工程策略对照。CPU 建立完整推理服务主线，单卡 CUDA 在保持上层语义不变的前提下替换执行后端，SmolVLM 多模态在解码器前扩展视觉输入并复用后续运行时，最终目标是构建一个可 DIY 的单机推理服务框架。

![cpp-llm-engine 教程主图](source/assets/diagrams/tutorial-concept-map.svg)

## Quick Start

第一次学习建议完成三个动作：在 Chapter00 看见一条完整的模型推理路径，在 Chapter01 读取模型配置和权重，再沿 [USAGE.md](USAGE.md) 进入 Tensor、算子、Decoder、Runtime 和服务章节。

先准备 CMake 3.20+、支持 C++17 的编译器和 Git。Linux 可以直接使用下面的命令；Windows 建议进入 WSL2 Ubuntu 后执行。安装方法和常见问题见 [Chapter00 环境准备](source/chapter00_quickstart/01_environment_setup.md)。

```bash
cmake --version
c++ --version
git --version
```

如果 C++ 所有权、CMake 或 CTest 仍然陌生，可以先从 [Foundation](source/foundation/intro.md) 选择对应练习；已经掌握这些基础时直接继续 Chapter00。

先运行不需要下载模型的 Chapter00 小实验。看到 CTest 显示测试通过，说明 C++ 编译和运行环境可以继续学习：

```bash
bash code/chapter00_quickstart/run.sh
```

真实推理所需的模型下载和路径设置见 [Chapter00 导读](source/chapter00_quickstart/intro.md)。暂时没有模型可以跳过下一条命令；CUDA 不是 CPU 学习路径的前置条件。

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

更新时间：**2026-11-06**。详细实验条件与复现入口见[实验资产表](source/reference/experiment_assets.md)。

| 时间 | 里程碑 | 状态 | 可验证产物 |
| --- | --- | --- | --- |
| 2026-08 | 教程骨架与 Foundation | 完成 | 章节主线、基础题库、benchmark trace |
| 2026-10 | CPU 主线闭环 | 完成 | TinyLlama 22 层 Decoder、Runtime、HTTP/SSE 和真实模型验收 |
| 2026-10 | 多模态输入闭环 | 完成 | SmolVLM Vision、Projector、Decoder、ServiceCore 和 profiling golden |
| 2026-11 | CUDA 后端 | 建设中 | CUDA 资源、量化 Linear、RoPE、KV、GQA 和合成 Decoder；下一步接真实模型 |
| 2026-11 | 工程验证体系 | 完成 | 默认回归、Release 契约、测试资产审计、文档检查和 GitHub Actions CI |

## 查阅与贡献

学习路线、章节命令和验证层级见 [USAGE.md](USAGE.md)；术语、公式、实验和框架源码坐标见 [Reference](source/reference/README.md)；修改、测试和提交方法见 [CONTRIBUTING.md](CONTRIBUTING.md)。GitHub Actions 持续检查文档、默认回归和发布契约。

测试从对象契约逐步扩展到组件协作、跨章主线和真实模型；性能实验建立在正确性验证通过之后。

![教程验证证据图](source/assets/diagrams/test-pipeline.svg)
