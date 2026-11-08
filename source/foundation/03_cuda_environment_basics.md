# F03 CUDA 与环境基础

CUDA 学习先解决“环境是否可用”和“CPU 结果如何作为基线”，再进入 kernel、cuBLAS 和异步执行。没有 GPU 时，仍然可以完成 CPU 主线。

## 驱动、工具链与设备

| 检查对象 | 需要确认什么 | 失败时先看哪里 |
| --- | --- | --- |
| Driver | 系统能发现 GPU | `nvidia-smi` 或设备权限 |
| Toolkit | `nvcc` 和 headers 可用 | CUDA 安装与 PATH |
| Runtime | 程序能枚举设备 | 动态库和运行时版本 |
| 构建开关 | target 确实编译 CUDA 文件 | CMake 配置和日志 |
| CPU fallback | 无 GPU 仍能跑正确性测试 | CPU reference |

## CPU 与 CUDA 的分工

| 路径 | 负责什么 | 本教程入口 |
| --- | --- | --- |
| CPU | 正确性、调试、回归和基线性能 | 默认路径，贯穿 Chapter 01-10 |
| CUDA | device memory、kernel 和加速对照 | FQ10、Chapter 03 以后 |
| 外部框架 | 展示成熟实现的选择 | llama.cpp、vLLM、TensorRT-LLM |

`GGML_CUDA=ON` 是 llama.cpp 的构建选项；教程自己的代码使用类似 `ENABLE_CUDA` 的 target 选项。学习重点是构建开关如何改变代码路径，而不是记住某个项目的变量名。

## 最小示例

```cpp
__global__ void add_kernel(const float* a, const float* b, float* c, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) c[i] = a[i] + b[i];
}
```

先写相同输入输出约定的 CPU reference，再比较 CUDA 结果。这样 kernel 错误可以和模型、tokenizer、服务层问题分开。

## 验证与练习

验证顺序：确认设备 -> 配置 CUDA target -> 编译 -> 枚举设备 -> 与 CPU 结果比较。对应练习见 [Foundation 正文与练习映射](./exercise_map.md)，完成 `FQ10`；它验证 kernel 映射，不替代完整环境排查。

仓库的第一条自研 CUDA 路径从 <a href="../_code/cuda/include/cpp_llm_engine/cuda/device_buffer.h"><code>DeviceBuffer</code></a> 开始。它独立于 CPU 默认回归，已经把错误传播、device/pinned host 内存、stream、event 和异步复制串成最小闭环。使用兼容 Toolkit 的宿主环境运行：

```bash
CUDA_HOME=/usr/local/cuda-12.8 bash code/cuda/scripts/run.sh
```

| 当前资产 | 学习重点 | 验证证据 |
| --- | --- | --- |
| `CudaStatus` | 保留 CUDA 错误码、操作名和可读信息 | 人工错误和 launch 错误能转成 core `Status` |
| `DeviceBuffer` | move-only 所有权、零长度和显式释放 | `cudaMalloc/cudaFree` 与移动后源对象变空 |
| `PinnedHostBuffer` | 页锁定 host 内存和可重叠复制前提 | `cudaMallocHost/cudaFreeHost`、移动和重复释放 |
| `CudaStream` / `CudaEvent` | 异步顺序、同步边界和 GPU elapsed time | 同一 stream 串联 H2D、kernel、D2H，并由 event 记录完成时间 |
| 同步/异步 H2D/D2H | host 指针、offset、容量和 stream 边界 | 真实 kernel 前后数据正确，无效句柄和越界在入队前拒绝 |
| `CudaElementwiseBackend` | grid/block、非整除边界和 backend 设备身份 | Vector Add 与 CPU golden 一致，固定 workload 用 event 多轮计时 |
