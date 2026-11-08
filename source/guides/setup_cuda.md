# CUDA 安装与配置

CPU 主线可以完成 Chapter00–10 的默认学习与验证。准备运行 `code/cuda`、真实 CUDA 模型或 GPU 性能实验时，再按本指南配置 NVIDIA 环境。

## CPU 与 GPU 环境分别支持什么

| 学习目标 | 所需环境 | 入口 |
| --- | --- | --- |
| 阅读正文、运行 CPU 章节和默认回归 | C++17 编译器、CMake、Git、Node.js | [使用说明](/guide) |
| 编译 CUDA Kernel、cuBLAS 和设备资源测试 | NVIDIA GPU、驱动、CUDA Toolkit | `code/cuda/scripts/run.sh` |
| 运行真实 TinyLlama CUDA Runtime 与服务 | 上述环境和 TinyLlama GGUF | `code/engine/scripts/run_real_cuda_runtime.sh` |

CUDA 是扩展路径的运行条件，不影响 CPU reference、接口设计和正确性基线。先通过 CPU 测试，再排查设备编译或数值差异，可以把代码问题与环境问题分开。

## 驱动、Toolkit 与 Runtime

| 组件 | 作用 | 检查入口 |
| --- | --- | --- |
| NVIDIA 驱动 | 让操作系统访问 GPU，并提供驱动接口 | `nvidia-smi` |
| CUDA Toolkit | 提供 `nvcc`、头文件、静态库和开发工具 | `nvcc --version` |
| CUDA Runtime | 程序运行时使用的设备管理、内存和 Kernel 启动接口 | 真实 CUDA 测试 |

三者需要兼容，但版本号含义不同。`nvidia-smi` 显示的 CUDA 版本表示驱动可支持的最高兼容级别，`nvcc --version` 才表示当前 Toolkit 编译器版本。

本仓库已使用 Ubuntu、GCC 13 和 CUDA 12.8 完成真实设备验收。其他支持 C++17 的 GCC/Clang 与 CUDA 12.x 组合也可以尝试，最终以配置、编译和设备测试结果为准。驱动与 Toolkit 的安装包应从 NVIDIA 对应 Linux 发行版渠道获取。

## Linux 配置顺序

先确认 GPU 和驱动：

```bash
nvidia-smi
```

再确认 Toolkit 与基础工具：

```bash
nvcc --version
cmake --version
g++ --version
git --version
```

机器安装了多个 Toolkit 时，为当前终端明确选择版本：

```bash
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
hash -r
```

`CUDA_HOME` 必须指向包含 `bin/nvcc` 和目标平台库目录的 Toolkit 根目录。Shell 配置生效后重新运行版本检查，确认 CMake 使用的是同一套 Toolkit。

Windows 推荐在 WSL2 Ubuntu 中使用仓库脚本。Windows 宿主机安装支持 WSL 的 NVIDIA 驱动，Toolkit、CMake 和编译器安装在 WSL 内；先在 WSL 终端确认 `nvidia-smi` 与 `nvcc`。

## 运行仓库验收

从独立 CUDA 组件开始，再进入真实模型 Runtime 和服务：

```bash
CUDA_HOME=/usr/local/cuda-12.8 bash code/cuda/scripts/run.sh
bash code/engine/scripts/run_real_cuda_runtime.sh
bash code/engine/scripts/run_real_cuda_serving.sh
```

第一条验证设备内存、Stream/Event、cuBLAS、量化 Linear、RoPE、K/V 和 Decoder；后两条验证 TinyLlama 22 层生成、分页缓存、同步 Completion、SSE 和资源释放。模型路径默认为 `llama.cpp/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf`，也可设置 `CPP_LLM_ENGINE_MODEL_PATH`。

只学习单个 CUDA 概念时，先阅读 [Foundation CUDA 环境基础](../foundation/03_cuda_environment_basics.md)；CPU/GPU 对象怎样共享上层契约见 [CPU/GPU 架构](../reference/architecture.md#cpu-主线与-gpu-扩展)。

## 常见故障

| 现象 | 说明 | 优先检查 |
| --- | --- | --- |
| `nvidia-smi` 找不到设备 | 驱动或设备访问尚未就绪 | 宿主机驱动、容器/WSL GPU 映射 |
| `nvidia-smi` 正常，`nvcc` 不存在 | Toolkit 未安装或 `PATH` 未选择正确版本 | `$CUDA_HOME/bin/nvcc` |
| CMake 找到另一套 CUDA | 环境中存在多个 Toolkit | `CUDA_HOME`、`PATH`、CMake 配置输出 |
| 编译通过，运行时报设备错误 | 设备架构、驱动、内存或参数存在问题 | 第一条失败的 CUDA CTest |
| CPU 与 CUDA 数值不同 | 环境已可运行，执行结果未满足契约 | shape、dtype、位置、K/V 和 CPU golden |
| 显存不足 | 当前模型、上下文或批量超过设备容量 | 模型量化、上下文长度、并发请求数 |

验收记录至少保存 GPU 型号、驱动、Toolkit、编译器、模型文件和失败测试名称。编译成功只证明工具链可用；真实设备执行与 CPU 数值对照通过后，才能确认当前 CUDA 路径成立。
