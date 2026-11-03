# 用 C++ 构建 LLM 推理引擎

_A Practical Tutorial for Building a C++ LLM Inference Engine_

这是一套中文优先、CPU 可完整学习、GPU 可选扩展的工程教程。你将从真实模型基线出发，逐步构建能够修改、测试和解释的 C++ 推理服务。

![cpp-llm-engine 教程主图](./assets/diagrams/tutorial-concept-map.svg)

## Quick Start

准备好 CMake、C++17 编译器、Node.js 和 npm 后，先运行无需模型文件的轻量回归：

```bash
npm install
npm run test:regression
bash code/chapter00_quickstart/run.sh
```

随后按 [Chapter 00](/chapter00_quickstart/intro) 准备 `llama.cpp` 和 TinyLlama GGUF，运行第一次真实推理。完整学习顺序与可选扩展见[阅读指南](/guide)。

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
