# F03 CUDA 与环境基础

CUDA 学习先解决“环境是否可用”和“CPU 结果如何作为基线”，再进入计算核（kernel）、cuBLAS 和异步执行。CPU 主线可以独立完成，GPU 路线在此基础上增加设备执行。

## 驱动、工具链与设备

| 检查对象 | 需要确认什么 | 失败时先看哪里 |
| --- | --- | --- |
| 驱动（Driver） | 系统能发现 GPU | `nvidia-smi` 或设备权限 |
| 工具包（Toolkit） | `nvcc` 和头文件可用 | CUDA 安装与 `PATH` |
| CUDA 运行时（CUDA Runtime） | 程序能枚举设备 | 动态库和运行时版本 |
| 构建开关 | 构建目标确实编译 CUDA 文件 | CMake 配置和日志 |
| CPU 路径 | CPU 正确性测试通过 | CPU 参考实现 |

## CPU 与 CUDA 的分工

| 路径 | 负责什么 | 本教程入口 |
| --- | --- | --- |
| CPU | 正确性、调试、回归和基线性能 | 默认路径，贯穿 Chapter 01-10 |
| CUDA | 设备内存、计算核和加速对照 | FQ10、Chapter03 以后 |
| 外部框架 | 展示成熟实现的选择 | llama.cpp、vLLM、TensorRT-LLM |

`GGML_CUDA=ON` 是 llama.cpp 的构建选项；教程自己的代码使用类似 `ENABLE_CUDA` 的构建目标选项。学习重点是构建开关如何改变代码路径，而不是记住某个项目的变量名。

## 最小示例

```cpp
__global__ void add_kernel(const float* a, const float* b, float* c, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) c[i] = a[i] + b[i];
}
```

先写相同输入输出约定的 CPU 参考实现，再比较 CUDA 结果。这样可以把计算核错误与模型、分词器、服务层问题分开。

## 验证与练习

验证顺序：确认设备 -> 配置 CUDA 构建目标 -> 编译 -> 枚举设备 -> 与 CPU 结果比较。对应练习见 [Foundation 正文与练习映射](./exercise_map.md)，完成 `FQ10`；它验证计算核映射，环境问题仍按本节表格排查。

仓库的第一条自研 CUDA 路径从 <a href="https://github.com/datawhalechina/cpp-llm-engine/blob/main/code/cuda/include/cpp_llm_engine/cuda/device_buffer.h"><code>DeviceBuffer</code></a> 开始。它独立于 CPU 默认回归，已经把错误传播、device/pinned host 内存、stream、event 和异步复制串成最小闭环。使用兼容 Toolkit 的宿主环境运行：

```bash
CUDA_HOME=/usr/local/cuda-12.8 bash code/cuda/scripts/run.sh
```

| 代码对象 | 学习重点 | 验证证据 |
| --- | --- | --- |
| `CudaStatus` | 保留 CUDA 错误码、操作名和可读信息 | 人工错误和启动错误能转成核心 `Status` |
| `DeviceBuffer` | move-only 所有权、零长度和显式释放 | `cudaMalloc/cudaFree` 与移动后源对象变空 |
| `PinnedHostBuffer` | 页锁定主机内存和可重叠复制前提 | `cudaMallocHost/cudaFreeHost`、移动和重复释放 |
| `CudaStream` / `CudaEvent` | 异步顺序、同步边界和 GPU 经过时间 | 同一 CUDA 流串联 H2D、计算核、D2H，并由事件记录完成时间 |
| 同步/异步 H2D/D2H | 主机指针、偏移、容量和 CUDA 流边界 | 真实计算核前后数据正确，无效句柄和越界在入队前拒绝 |
| `CudaElementwiseBackend` | 网格/线程块、非整除边界和后端设备身份 | 向量加法与 CPU 基准结果一致，固定工作负载用事件多轮计时 |
