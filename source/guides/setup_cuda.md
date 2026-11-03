# CUDA Setup CUDA 安装与配置

本文档说明如何准备本教程需要的 CUDA 环境。目标不是覆盖所有发行版细节，而是给出一条清晰、可验证、可回退的最小路径。

## 1. 适用范围

- 你准备在本教程中使用 GPU 路径
- 你需要编译或运行带 CUDA 支持的章节
- 你想先确认驱动、工具链和运行时是否可用

CPU 路径可以直接从 `chapter00_quickstart` 和基础章节开始；进入 `code/cuda` 前再安装并验证 CUDA。

## 2. 版本建议

下面是一组适合本教程的最小建议版本。先用它们确认编译器、驱动和 runtime 能配合工作，不绑定某一个发行版。

| 组件 | CPU 路径建议 | CUDA 路径建议 | 说明 |
| --- | --- | --- | --- |
| 操作系统 | Ubuntu 22.04 LTS | Ubuntu 22.04 LTS | 本教程以 Linux 为主，优先推荐 22.04 LTS |
| CMake | 3.20+ | 3.20+ | 仓库内所有章节的 `CMakeLists.txt` 都以 3.20 为基线 |
| C++ 编译器 | GCC 11+ 或 Clang 14+ | GCC 11+ 或 Clang 14+ | 只要支持 C++17 即可 |
| Git | 2.34+ | 2.34+ | 用于拉取源码和管理教程内容 |
| NVIDIA 驱动 | 不需要 | 与所选 CUDA Toolkit 兼容 | 驱动和 Toolkit 要按官方兼容矩阵匹配 |
| CUDA Toolkit | 不需要 | 12.1+（推荐 12.x 系列） | 如果你要走 GPU 路径，再安装 |

如果你已经有一套稳定的环境，且版本高于上面建议，一般也可以继续使用；只要后续 `cmake`、编译和运行验证都能通过，就不需要为了版本号而盲目降级。

## 3. 先理解三个概念

- `driver`：负责让操作系统和 GPU 通信
- `toolkit`：提供 `nvcc`、头文件和 CUDA 库，用来编译 CUDA 代码
- `runtime`：程序运行时实际使用的 CUDA 相关组件

常见错误是把 `driver` 和 `toolkit` 混为一谈。它们不是同一个东西。

## 4. 推荐顺序

1. 先确认机器本身能看到 GPU
2. 再安装或确认 NVIDIA 驱动
3. 再安装 CUDA Toolkit
4. 最后验证编译器和运行时

如果某一步失败，先停下，不要继续往下叠加问题。

## 5. Linux 下的最小检查

### 5.1 检查 GPU 是否可见

```bash
nvidia-smi
```

如果命令能返回 GPU 信息，说明系统至少能看到显卡和驱动层。

### 5.2 检查 CUDA 编译器

```bash
nvcc --version
```

如果这个命令可用，说明 Toolkit 的编译器部分已经就绪。

### 5.3 检查基础构建工具

```bash
cmake --version
g++ --version
git --version
```

如果这些命令都可用，说明后续编译教程代码的基础工具已经具备。

## 6. CPU 先行原则

本教程不要求 CUDA 一开始就必须可用。

- 如果你当前只是学习主线，先把 CPU 路径跑通
- 如果你当前只是验证代码结构，先把编译和测试链路跑通
- 如果你后面要对比 GPU 版本，再回头补 CUDA 配置
- 当前教程主线没有任何章节把 CUDA 作为绝对前置条件
- `chapter00`、`chapter01`、`chapter02`、`chapter04`、`chapter06`、`chapter05`、`chapter07` 都可以先用 CPU 路径建立可验证基线

更明确地说：

- CPU 足够做的事情：
  - 读懂教程主线
  - 编译和运行基础章节
  - 验证张量形状、数据流和大多数正确性检查
  - 做算子接口、decoder 结构和生成流程的逻辑验证
- 一定要 CUDA 的事情：
  - 跑 GPU 加速的矩阵乘法或注意力实现
  - 做 CPU vs GPU backend 对照
  - 编译和运行显式依赖 CUDA backend 的章节
  - 使用 cuBLAS 或其他 CUDA 库做加速实验

这样做的好处是：

- 更快得到第一个可运行结果
- 更容易定位问题属于代码、编译还是 GPU 环境
- 避免把环境问题和教程内容问题混在一起

## 7. CMake 侧的典型用法

具体开关会随章节代码变化，但基本思路是一样的：

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
# 或者在支持 CUDA 的章节中启用相应开关
cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON
```

如果某个章节使用不同的选项名，请以该章节自己的 `README` 和 `CMakeLists.txt` 为准。

## 8. 常见问题

- `nvidia-smi` 能用，但 `nvcc` 不存在
  - 通常表示驱动有了，但 Toolkit 还没装好
- `nvcc` 能用，但编译时报找不到 CUDA 库
  - 通常表示库路径或环境变量没配好
- 编译能过，但运行时找不到 GPU
  - 通常表示驱动、运行时或容器配置有问题
- 本机没有 GPU
  - 先走 CPU 路径，不要卡住学习进度

## 9. 和本教程的关系

- `chapter00_quickstart`：优先先用 CPU 路径跑通
- `foundation/03_cuda_environment_basics.md`：理解 driver、toolkit、runtime 的区别
- `chapter03_matmul` 之后的章节：才更频繁地接触 CUDA 加速路径与性能优化概念

## 10. 常见章节的 CUDA 需求

为了避免“所有章节都必须先装好 CUDA”的误解，这里再明确一次：

- `chapter00_quickstart`：CPU 就够
- `chapter01_tensor_loader`：CPU 就够
- `chapter02_tensor_ops`：CPU 就够
- `chapter03_matmul`：CPU 能学会接口和基线，CUDA 用于加速对比
- `chapter04_attention`：CPU 能学会结构，CUDA 不是硬前置
- `chapter06_generation`：CPU 能跑通生成链路，CUDA 用于后续优化
- `chapter05_quantization`：CPU 可以理解原理，GPU 场景更适合做性能对照
- `chapter07_benchmark`：CPU 可以完成正确性与基础对照，GPU 用于性能评估

## 11. 哪些地方必须要 CUDA

当前仓库的主线章节里，没有任何一章把 CUDA 作为硬性前置条件。换句话说：

- 你可以在没有 CUDA 的机器上完成主线阅读和大部分验证
- 你可以先把 CPU 版本跑通，再考虑 GPU 加速
- 只有当你要做 GPU 加速实验、对照 GPU 路径，或者尝试 CUDA 相关优化时，才需要真正启用 CUDA

现有 CUDA 实现集中在 `code/cuda`，通过独立脚本显式构建和运行；Chapter03、Chapter05 的 CPU 默认脚本不会自动启用它。后续 Attention、device KV 和 GPU Decoder 也应继续在对应 README 与章节导读中标出运行条件。

## 12. 验证清单

在继续往下学之前，建议确认：

- `nvidia-smi` 可用
- `nvcc --version` 可用
- `cmake --version` 可用
- `g++ --version` 或 `clang++ --version` 可用
- 你知道自己当前是 CPU 路径还是 GPU 路径

编译并跑通 `chapter00` 和 `chapter01` 后即可继续 CPU 主线；GPU 验证条件可以在进入 CUDA 扩展时补齐。

## 13. English Summary

This document explains the minimum CUDA environment needed by the tutorial.
The key idea is to separate the driver, toolkit, and runtime, then validate
the setup step by step.

Recommended order:

1. Check that the GPU is visible
2. Install or verify the NVIDIA driver
3. Install the CUDA Toolkit
4. Verify `nvcc`, `cmake`, and the compiler toolchain

If you only want to follow the CPU path first, you can postpone CUDA setup
and return to it later when you need GPU acceleration.
