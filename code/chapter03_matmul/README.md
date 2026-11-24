# Chapter 03：矩阵乘法与计算后端

本目录接收教学 `Matrix` 或 Chapter02 张量视图，把二维矩阵乘转换为公共 `MatmulRequest`，再交给 CPU 参考后端执行。这里建立的形状检查、乘加公式和验证方法会继续用于 Chapter04 解码器；Chapter04 采用独立的教学线性层实现。原理、公式和设计推导从[Chapter03 导读](../../source/chapter03_matmul/intro.md)开始阅读。

## 代码架构

代码有两个输入入口。`MatMulOp` 接收拥有存储的教学 `Matrix`；`tensor_matmul_adapter` 接收借用 Chapter02 存储的 `TensorViewRef`。两条路径都要求 CPU、Float32、二维、连续行主序布局，并转换成借用缓冲区的 `MatmulRequest`。

`MatMulOp` 的构造函数接收计算后端的左值引用，使后端生命周期覆盖算子调用。CPU 后端报告设备身份并分派请求，三重循环位于参考内核。算子路径先创建临时 `Matrix`，视图路径先写入临时 `vector`；后端成功后再提交输出，因此形状检查或计算失败会保留调用方原结果。

```text
教学 Matrix ──> MatMulOp::try_compute ───────────────┐
                                                     ├─> MatmulRequest
Chapter02 Tensor View ──> tensor_matmul_adapter ─────┘         │
                                                               ▼
                                                  MatmulBackend 接口
                                                               │
                                                               ▼
                                               CPU 参考内核（三重循环）
                                                               │
                                                               ▼
                                                    临时结果 -> 成功后提交
```

![Chapter03 算子、计算后端与内核架构](../../source/assets/diagrams/chapter03-operator-backend-kernel.svg)

### 设计分层

| 层次 | 代码对象 | 职责 |
| --- | --- | --- |
| 输入与拥有对象 | `Matrix`、Chapter02 `Tensor` | 保存教学数据或共享存储 |
| 非拥有请求 | `TensorViewRef`、`MatmulRequest` | 描述地址、形状、布局和输出空间 |
| 算子与适配 | `MatMulOp`、`tensor_matmul_adapter` | 校验输入、构造请求、成功后提交结果 |
| 后端 | `CpuReferenceMatmulBackend`、CUDA cuBLAS 接口 | 管理设备相关分派和状态 |
| 内核或计算库 | `cpu_reference_matmul_kernel`、`cublasSgemm` | 执行实际乘加 |

| 执行路径 | 内存位置 | 接口特点 | 当前用途 |
| --- | --- | --- | --- |
| CPU 参考实现 | 主机 | 实现核心 `MatmulBackend` | 教学主线与跨后端数值基线 |
| CUDA cuBLAS | 设备 | 独立接口显式接收 CUDA 流 | GPU 正确性和性能基线 |
| 自定义 CUDA 计算内核 | 设备 | 复用 `MatmulRequest` 的形状与地址 | 融合算子的扩展入口 |

## 阅读顺序与资产

先从教学数据对象和公共请求开始，再沿算子、后端、内核进入计算，最后阅读 Tensor 适配和测试。这样可以先看清数据要求，再判断每个对象为什么存在。

| 顺序 | 文件 | 阅读重点 | 对应验证 |
| --- | --- | --- | --- |
| 1 | `include/chapter03_matmul/matrix.h`、`src/matrix.cpp` | 行主序存储、二维索引和尺寸溢出 | `chapter03_matmul_test` |
| 2 | `../core/include/cpp_llm_engine/core/matmul_contract.h` | `MatmulRequest` 的形状、缓冲区和后端接口 | `cpp_llm_engine_core_test` |
| 3 | `include/chapter03_matmul/matmul_op.h`、`src/matmul_op.cpp` | 形状检查、后端生命周期和成功后提交 | `chapter03_matmul_test` |
| 4 | `src/cpu_backend.cpp`、`src/cpu_kernel.cpp` | 后端分派与三重循环参考计算 | `chapter03_matmul_test` |
| 5 | `src/tensor_matmul_adapter.cpp` | 视图布局检查、偏移和显式连续化要求 | `chapter02_tensor_matmul_integration_test` |
| 6 | `examples/main_matmul.cpp`、`tests/` | 最小输出、失败路径和跨章证据 | `scripts/run.sh` |

## 运行与验证

从仓库根目录执行下面的命令。脚本使用隔离构建目录，先运行示例，再执行本章、跨章和 Core 测试。

```bash
bash code/chapter03_matmul/scripts/run.sh
```

示例应输出 `2x2`，左上角元素为 `58`，右下角元素为 `154`。三个 CTest 分别检查局部计算、Chapter02 视图交接和核心公共对象。

| 验证层级 | target | 证明内容 |
| --- | --- | --- |
| 组件 | `chapter03_matmul_test` | 固定数值、形状、溢出、后端生命周期和失败后保留输出 |
| 集成 | `chapter02_tensor_matmul_integration_test` | 连续与带偏移视图、显式连续化、接口错误和失败后保留输出 |
| 公共接口 | `cpp_llm_engine_core_test` | 计算后端、状态和张量公共数据结构 |

## 扩展入口

CPU 参考内核提供数值基线。cuBLAS 在独立 CUDA 构建目标中复用 `MatmulRequest` 的形状数据，并通过带 CUDA 流的接口执行；Chapter04 复用矩阵乘公式和公共张量视图，组合注意力与前馈网络。

| 方向 | 代码位置 | 连接对象 | 验证入口 |
| --- | --- | --- | --- |
| CUDA/cuBLAS | `code/cuda/src/cublas_matmul_backend.cpp` | `MatmulRequest`、设备缓冲区和 CUDA 流 | `CUDA_HOME=... bash code/cuda/scripts/run.sh` |
| 解码器 | `code/chapter04_attention/` | 线性层、注意力和前馈网络 | `bash code/chapter04_attention/scripts/run.sh` |
| 量化 GEMV | `code/chapter05_quantization/`、`code/cuda/` | 打包权重、融合反量化和小批量执行 | `bash code/chapter05_quantization/scripts/run.sh` |
| 批执行与性能决策 | `code/chapter08_scheduling/`、`code/chapter09_performance/` | 请求组批、实现选择和性能证据 | 学到对应章节后执行 |
