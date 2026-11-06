# 用 C++ 构建 LLM 推理引擎

_A Practical Tutorial for Building a C++ LLM Inference Engine_

这是一套中文优先、CPU 可完整学习、GPU 可选扩展的工程教程。你将从真实模型基线出发，逐步构建能够修改、测试和解释的 C++ 推理服务。

![cpp-llm-engine 教程主图](./assets/diagrams/tutorial-concept-map.svg)

## Quick Start

Quick Start 依次完成仓库检查、Chapter00 真实基线和 Chapter01 自研主线入口。Chapter02–10 的命令见[阅读指南](/guide)。

CPU 路径需要 CMake 3.20+、必须支持 C++17 的编译器和 Git；全仓库回归与文档工具另需 Node.js 和 npm。Linux 可直接运行，Windows 推荐使用 WSL2 Ubuntu。先阅读 [Chapter00 环境准备](/chapter00_quickstart/01_environment_setup)；CUDA 不是主线前置，需要时再阅读 [CUDA 环境指南](/guides/setup_cuda)。

如果还不熟悉 C++ 所有权、CMake 或 CTest，先按需学习 [Foundation](/foundation/intro)。Foundation 是通用能力和练习区，基础已具备时可以跳过。

```bash
cmake --version
c++ --version
node --version
npm install
npm run test:regression
```

再建立真实基线并进入自研主线：

```bash
bash code/chapter00_quickstart/run.sh
bash code/chapter00_quickstart/run_real_inference.sh  # 准备好 llama.cpp 和 TinyLlama 后运行
bash code/chapter01_tensor_loader/run.sh
```

真实推理脚本依赖 [Chapter 00](/chapter00_quickstart/intro) 中准备的 `llama.cpp` 和 TinyLlama GGUF。

## 学习产物

| 阶段 | 核心对象 | 完成后的产物 |
| --- | --- | --- |
| 真实基线 | TinyLlama、GGUF、Tokenizer | 可复现的本地推理输出 |
| 引擎对象 | Model、Tensor/View、Operator、Backend | 可测试的 CPU 执行底座 |
| 模型与 Runtime | Decoder、Request、KV Cache、Scheduler | 可推进单请求和多请求的生成运行时 |
| 验证与服务 | Benchmark、Profiling、ServiceCore、HTTP/SSE | 可复现的最小 CPU 推理服务 |
| 可选扩展 | CUDA、SmolVLM 多模态输入 | GPU Backend 或视觉输入扩展 |

## 教程特点

| 特点 | 提供的证据 |
| --- | --- |
| C++ 可运行主线 | 章节实现、CTest 和统一 CPU 回归 |
| Correctness-first | reference、golden、benchmark 和 profiling |
| 真实模型纵向链路 | TinyLlama 与 SmolVLM 的 token、logits、checksum 和服务验证 |
| 成熟框架对照 | 章节对象映射与 ecosystem 项目卡 |

## 当前进度

| 路线 | 状态 | 当前能力 |
| --- | --- | --- |
| CPU 主线 | 已闭环 | TinyLlama 22 层 Decoder、Generation Runtime 和 HTTP/SSE 服务 |
| 多模态输入 | 已闭环 | SmolVLM Vision、Projector、Decoder 和 ServiceCore |
| CUDA 后端 | 建设中 | CUDA 资源、量化 Linear、RoPE、KV、GQA 和合成 Decoder |
