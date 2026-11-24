# 03.2 矩阵乘法请求怎样执行

两个矩阵能够相乘，需要满足 `lhs.cols == rhs.rows`。代码还要知道数据地址、行优先步长和输出位置，才能把数学公式变成一次安全的内存访问。

本章线性层、矩阵乘法和权重转置统一采用 [`F-LINEAR-01`](../reference/formulas.md#f-linear-01) 的形状约定。

## 先固定输入和输出形状

本章使用连续行优先布局：

```text
左输入 [M,K]，步长 [K,1]
右输入 [K,N]，步长 [N,1]
输出   [M,N]，步长 [N,1]
```

二维视图中第 `row` 行、第 `col` 列的元素地址为：

```text
storage[offset + row * stride[0] + col * stride[1]]
```

连续行优先矩阵的步长是 `[cols,1]`，地址可以简化为 `offset + row * cols + col`。CPU 内核正是按这个公式读取元素，因此转置或跨步切片要先整理成连续副本。

## GEMV 和 GEMM 描述什么形状

GEMV 是 General Matrix-Vector Multiplication，即通用矩阵向量乘法；GEMM 是 General Matrix-Matrix Multiplication，即通用矩阵矩阵乘法。二者使用相同的乘加关系，区别在于右侧输入是一条向量还是多行矩阵。

| 推理场景 | 典型输入 | 计算称呼 | 主要复用机会 |
| --- | --- | --- | --- |
| 单词元解码 | `x[K]` | GEMV：`y = W x` | 一次读取权重并完成一个向量 |
| 多词元预填充 | `X[T,K]` | GEMM：`Y = X Wᵀ` | 多个词元复用同一权重分块 |
| 小批量解码 | `X[B,K]` | 数学上是 GEMM；专用实现常称批量 GEMV | 多个请求共享权重读取与解码 |

本章公共 `MatMulOp` 接收二维矩阵，调用方显式传入 CPU 计算后端，CUDA 路径使用独立的 cuBLAS 接口。Chapter05 的量化融合路径称为批量 GEMV：它扩展单向量计算内核，让少量输入行共享一次打包权重解码。Chapter08 决定哪些请求进入同一执行批次，Chapter09 再用固定负载比较不同实现。

每个输出元素由左输入的一行和右输入的一列计算。`try_compute(lhs, rhs, &output)` 先检查矩阵对象和形状，再把地址与 `M、N、K` 写入 `MatmulRequest`；`compute()` 使用同一流程，在失败时抛出异常。

## 从张量视图到输出写回

<a href="https://github.com/datawhalechina/cpp-llm-engine/blob/main/code/chapter03_matmul/src/tensor_matmul_adapter.cpp"><code>tensor_matmul_adapter.cpp</code></a> 按下面的顺序处理 Chapter02 视图：

| 步骤 | 代码检查或操作 | 目的 |
| --- | --- | --- |
| 1 | 检查存储范围、二维形状、Float32 和 CPU 设备 | 确认地址和元素解释有效 |
| 2 | 检查连续行优先步长 | 让内核可以使用 `row * cols + col` 寻址 |
| 3 | 基址加 offset，创建 `MatmulRequest` | 指向视图中的第一个参与计算元素 |
| 4 | 在临时连续结果中运行 CPU 计算后端 | 失败时保留调用者原来的输出 |
| 5 | 计算成功后复制到 `MutableTensorViewRef` | 一次提交完整结果 |

切片或转置产生的非连续视图会返回 `matmul requires contiguous row-major input`。调用 `contiguous()` 得到连续副本后，可以重新提交计算。

## 一次调用经过哪些代码

| 位置 | 当前实现 | 阅读重点 |
| --- | --- | --- |
| 算子入口 | `MatMulOp` | 接收输入并创建请求 |
| CPU 计算后端 | `CpuReferenceMatmulBackend` | 检查请求并选择 CPU 内核 |
| CPU 内核 | `cpu_kernel.cpp` | 用三重循环计算每个输出元素 |
| GPU 选读 | `CublasMatmulBackend` | 用设备内存和 `cublasSgemm` 完成同一形状的计算 |

CPU 和 CUDA 的输入地址位于不同设备。行优先数据交给 cuBLAS 时还要处理列优先解释，具体步骤放在 [`03.4`](./04_cublas_implement.md)。

## 临时结果和工作区由谁持有

输出先写入临时结果，计算后端成功后才覆盖调用者的张量。工作区保存计算期间的中间数据，例如量化权重解码结果或 GPU 归约空间；二者都不改变输入张量的所有权。

工作区容量先由形状计算并校验，仅在空间不足时增长。异步 GPU 执行需要等待使用它的 CUDA 流或事件完成后再复用。CPU 参考实现提供相同的输入、输出和错误行为，设备实现用它做数值对照。GPU 资源所有者见 <a href="https://github.com/datawhalechina/cpp-llm-engine/blob/main/code/cuda/include/cpp_llm_engine/cuda/cuda_workspace.h"><code>CudaWorkspace</code></a> 和 <a href="https://github.com/datawhalechina/cpp-llm-engine/blob/main/code/cuda/include/cpp_llm_engine/cuda/cuda_execution_context.h"><code>CudaExecutionContext</code></a>。

## 动手验证

```bash
bash code/chapter03_matmul/scripts/run.sh
```

测试覆盖合法矩阵、非零偏移视图、显式连续化，以及空引用、错误类型、越界、非连续布局和形状不匹配。所有输入检查通过后，再进入 [`03.3`](./03_cpu_reference.md) 阅读实际计算循环。
