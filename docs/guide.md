# 阅读指南

默认沿 CPU 路径学习，在稳定接口和正确性基线上继续选择 GPU 或多模态扩展。

## 入口结构

教程按依赖关系排列为基础准备、引擎实现、验证优化和服务扩展：

```text
开始这里
  Foundation -> Chapter 00 Quick Start

引擎主线
  Chapter 01 -> Chapter 06

验证与优化
  Chapter 07 -> Chapter 08

服务扩展与参考
  Chapter 09（Chapter06 后可提前阅读）-> reference / guides
```

各部分提供以下学习产物：

- `foundation`：补 C++、CMake、CUDA、调试和测试基础
- `chapter00_quickstart`：用 `llama.cpp` 完成一次真实推理，确认环境和模型链路成立
- `chapter01` - `chapter06`：逐层构建 Tensor、算子、Decoder、量化 Linear 和 Runtime
- `chapter07` - `chapter08`：用回归、Benchmark 和 Profiling 形成证据闭环
- `chapter09`：接收 Chapter06 的服务化扩展，当前包含 C++ Service Core、metrics、真实模型 HTTP/SSE completion 和可选 cpp-httplib loopback
- `reference` / `guides`：查阅概念、实验资产和工程方法

如果你想先理解“训练框架、推理框架和本教程分别负责什么”，再看 [训练框架 vs 推理框架](/reference/framework_comparison)。`foundation` 是按需前置能力区，`chapter00` 是所有学习者共用的真实推理起跑线。

## 从这里开始

需要按可验证产物控制阅读量时，使用 [CPU 六任务学习计划](/guides/phase1_cpu_learning_plan)中的“首轮与进阶范围”；第一次学习只完成首轮必读和主验证命令。

1. 先看 [教程主入口](/README) 了解总图
2. 需要 C++、构建工具或 CUDA 背景时，按需进入 `foundation`
3. 进入 `chapter00_quickstart`，用 `llama.cpp` 完成 First Success
4. 按 `chapter01` 到 `chapter06` 学习引擎核心
5. 进入 `chapter07` 和 `chapter08` 做验证与性能归因
6. 完成 Chapter06 后，如果关注服务化，可以提前进入 Chapter09；否则先完成 Chapter07-08 的 Benchmark 和 Profiling，再回到 Chapter09

## 如何运行代码

优先使用各章节目录下的脚本：

- `bash code/chapter00_quickstart/run.sh`
- `bash code/chapter02_tensor_ops/scripts/run.sh`
- `bash code/chapter03_matmul/scripts/run.sh`
- `bash code/chapter04_attention/scripts/run.sh`
- `bash code/chapter05_quantization/scripts/run.sh`
- `bash code/chapter06_generation/scripts/run.sh`
- `bash code/chapter07_benchmark/scripts/run.sh`
- `bash code/chapter08_profiling/scripts/run.sh`
- `bash code/chapter09_serving/scripts/run.sh`

对于 `chapter01_tensor_loader`，使用隔离构建脚本：

```bash
bash code/chapter01_tensor_loader/run.sh
```

CUDA 与多模态使用独立入口，CPU 回归仍由主线脚本验证：

```bash
bash code/cuda/scripts/run.sh
bash code/multimodal/scripts/run.sh
```

## 硬件与环境

这套教程以 CPU 为默认环境，并提供本地 GPU 和云端 GPU 两种扩展环境。

### 1. 本地 CPU 学习环境

本地 CPU 环境可以完整跑通 `foundation` 和 Chapter00-09 的默认路径。

建议配置：

- 4 核以上 CPU
- 16 GB 内存
- 20 GB 以上可用磁盘
- Linux 环境，优先 Ubuntu 22.04
- `CMake`、`g++` 或 `clang++`、`git`
- `Node.js + npm`

### 2. 本地带 GPU 的学习环境

当前 GPU 路径可以运行 `code/cuda` 的设备资源、cuBLAS MatMul、Q4_K/Q6_K Linear、RoPE、连续 KV、GQA 和合成 Decoder；下一里程碑是真实 TinyLlama GPU Decoder。

建议配置：

- 单张 NVIDIA 消费级 GPU
- 首期以 TinyLlama Q4 在 8 GB 显存内完成端到端运行为设计目标
- 最低显存等待 CUDA 路径完成后，通过 4/6/8 GB 限额或设备实测确定
- 12 GB 及以上只为长上下文、更多请求或更大模型提供扩展余量
- 16 GB 以上内存更稳
- SSD
- NVIDIA Driver + CUDA Toolkit
- `nvcc`、`nvidia-smi` 可用

`8 GB` 不是最低门槛，`12 GB` 也不是必需配置。具体的 CUDA 安装与验证步骤见 [CUDA 安装与配置](/guides/setup_cuda)，显存口径和测量要求记录在仓库根 README 的“GPU 定位与显存口径”。

CPU 和 GPU 的环境差异，以及它们在章节中的分叉位置：

![CPU/GPU 学习路线与环境](./assets/diagrams/cpu-gpu-learning-route.svg)

### 3. 云端环境

适合本地没有合适 GPU，或者只想把 `chapter03` 之后的 GPU 对照和 benchmark 放到云上跑。

建议做法：

- 选 Linux 镜像，优先 Ubuntu 22.04
- 选带 NVIDIA GPU 的实例
- 先确认 `nvidia-smi`
- 再确认 `nvcc --version`
- 再装 `cmake`、`g++`、`git`、`node`
- 用持久盘保存仓库和模型文件

### 最小建议顺序

1. 先装基础工具链
2. 先跑 CPU 路径
3. 需要 GPU 时再补本地 CUDA 或云端 GPU
4. 用 `nvidia-smi`、`nvcc --version`、`cmake --version` 做最小验证

## CPU / GPU 分界线

- `foundation`、`chapter00`、`chapter01`、`chapter02`：CPU-first 就够
- `chapter03_matmul`：先固定 CPU backend 契约，再说明 GPU backend 怎样接入
- `chapter03_matmul/04_cublas_implement.md`：cuBLAS 接入与 row-major 映射；可构建实现位于 `code/cuda`
- 软件环境说明：看 [CUDA 安装与配置](/guides/setup_cuda)

代码架构图展示公共契约、章节库、Engine 集成和服务扩展之间的关系；测试流水线图串联单章 CTest、章节脚本、全局回归和文档检查。

![代码架构与测试流水线](./assets/diagrams/code-architecture.svg)

![测试流水线](./assets/diagrams/test-pipeline.svg)

## 按需补充

主线从 Chapter 00 走到 Chapter 09。基础薄弱时先补 Foundation；已有推理系统经验时可以在完成 Chapter 06 后提前阅读 Runtime、Benchmark、Profiling 和 Serving，并用对应章节脚本确认实现。

| 情况 | 补充内容 | 验证 |
| --- | --- | --- |
| C++、CMake 基础不稳 | Foundation 必读题和至少 4 道练习 | `bash exercises/foundation/run_all.sh` |
| 关注 Runtime 与服务 | Chapter 06、07、08、09 | 章节 `run.sh` 与 `npm run test:regression` |
| 关注框架实现差异 | `reference/ecosystem` 与各章对照页 | 回到本教程代码确认对应契约 |
| 需要 GPU | Chapter 03 CUDA 设计与 `code/cuda` | `bash code/cuda/scripts/run.sh`，完成 CPU/CUDA 数值对照和设备测试 |

## 开源项目怎么配合读

教程先用最小实现解释引擎原理，再用开源项目定位同一概念的生产实现。

建议按章节配合这些项目阅读：

- `chapter00`、`chapter01`：先看 `llama.cpp`，建立最小本地推理直觉
- `chapter02`、`chapter03`：再看 `gemma.cpp` 和 `TensorRT-LLM`，理解对象、算子和后端分层
- `chapter04`、`chapter06`：再看 `vLLM`、`SGLang`，理解 runtime、调度和 serving 组织方式
- `chapter05`：再看 `TensorRT-LLM`，理解量化和部署如何接到 GPU engine
- `chapter07`：再看 `vLLM`、`SGLang`、`Ollama`，理解 benchmark、验证和交付怎么落到真实系统
- `chapter08`、`chapter09`：再看 `vLLM`、`SGLang` 和 DeepInfer，理解 profiling、服务边界与生产扩展

如果你想先看总览，直接进 [生态对照](/reference/ecosystem/README)。

## 日常检查

不同命令证明的范围不同。默认回归覆盖无需模型下载、真实 socket 和 GPU 的轻量路径：

```bash
npm run docs:check-structure
npm run docs:check-links
```

再跑：

```bash
npm run code:check-headers
npm run docs:build
npm run test:regression
```

| 验证层级 | 当前入口 | 覆盖范围 |
| --- | --- | --- |
| 默认 CPU | `npm run test:regression` | Foundation、Core、Engine 和 Chapter01-09 轻量路径 |
| 真实模型 | `code/engine` 中设置 `CPP_LLM_ENGINE_REAL_MODEL` 的 CTest | TinyLlama golden、Context 隔离、HTTP adapter、profiling |
| 真实 transport | `run_cpp_httplib_loopback.sh` | loopback HTTP、metrics、同步与 SSE |
| 性能实验 | Chapter07-08 与 Engine benchmark 脚本 | 固定 workload 的结果文件和 A/B 判定 |
| GPU | `bash code/cuda/scripts/run.sh` | CUDA 资源、算子、KV、GQA 和合成 Decoder；不包含真实模型端到端生成 |
