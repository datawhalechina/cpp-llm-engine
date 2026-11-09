# 阅读指南

默认先完成 CPU 主线，再按学习目标选择 CUDA 或多模态。每章先读导读和正文，再进入代码 README、运行脚本和测试。

## 从这里开始

1. 在 [Chapter00](/chapter00_quickstart/intro) 用 llama.cpp 和 TinyLlama 建立真实推理基线。
2. C++、CMake 或测试基础不足时，按需进入 [Foundation](/foundation/intro)。
3. 按 Chapter01–10 学习模型输入、执行、运行时、缓存、调度、性能和服务。

| 阶段 | 章节 | 学习产物 |
| --- | --- | --- |
| 输入与执行底座 | Chapter01–03 | Model、Tensor/View、Operator 和 Backend |
| 模型与生成 | Chapter04–06 | Decoder、量化 Linear、Request 和 Generation Runtime |
| 缓存与调度 | Chapter07–08 | 分页 K/V、Radix Cache、Scheduler 和 Tensor batch |
| 证据与服务 | Chapter09–10 | Benchmark、Profiling、ServiceCore 和 HTTP/SSE |

需要按阶段产物推进时使用 [CPU 六任务学习计划](/guides/cpu_learning_plan)。章节正文解释原理和设计，`code/<chapter>/README.md` 说明源码架构、阅读顺序和验证入口。

## 选择扩展路线

| 路线 | 前置条件 | 入口 |
| --- | --- | --- |
| CPU 主线 | C++17、CMake 3.20+、Git | [CPU 六任务学习计划](/guides/cpu_learning_plan) |
| GPU 扩展 | CPU 执行契约、NVIDIA Driver 和 CUDA Toolkit | [GPU 六任务学习计划](/guides/gpu_learning_plan) |
| 多模态扩展 | Decoder 与 Runtime 基线 | [多模态六任务学习计划](/guides/multimodal_learning_plan) |

Linux 可直接使用仓库脚本；Windows 推荐 WSL2 Ubuntu。CUDA 不是 CPU 主线的前置条件，完整环境检查见 [Chapter00 环境准备](/chapter00_quickstart/01_environment_setup)。

## 运行与验证

各章使用对应 `code/<chapter>/run.sh` 或 `scripts/run.sh`。第一次进入仓库先运行无需模型和 GPU 的默认回归：

```bash
npm install
npm run test:regression
```

默认回归证明轻量主线没有退化，但不替代真实 GGUF、CUDA 设备、网络服务或性能实验。术语和公式见 [Reference](/reference/README)，跨章节环境和复现流程见 [Guides](/guides/README)。
