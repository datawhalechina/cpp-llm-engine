# 03.2 矩阵乘法算子设计

矩阵乘法的数学公式、内存位置和执行实现必须分开描述。`MatMulOp` 固定主机输入上的 `A[M,K] × B[K,N] = C[M,N]`；独立 CUDA 目标复用相同形状语义，但接收设备指针和 CUDA 流。

## 输入输出与布局契约

Chapter03 当前接受连续行优先 `Matrix`：

```text
lhs [M,K] stride [K,1]
rhs [K,N] stride [N,1]
out [M,N] stride [N,1]
```

`try_compute(lhs, rhs, &output)` 检查对象是否有效、`lhs.cols == rhs.rows` 和输出元素数量，再创建公共 `MatmulRequest`。教学入口 `compute()` 复用同一逻辑，并在失败时抛出异常。

工程入口 <a href="../_code/chapter03_matmul/src/tensor_matmul_adapter.cpp"><code>tensor_matmul_adapter.cpp</code></a> 接收两个只读 `TensorViewRef` 和一个可写 `MutableTensorViewRef`，依次检查存储范围、维数、Float32、CPU、连续行优先布局和输入输出形状。连续性由公共 `TensorContract` 判断，Chapter03 不再另写一套布局规则。验证通过后，适配器把存储基址加上偏移，生成指向视图首元素的 `MatmulRequest`。非连续切片或转置返回 `matmul requires contiguous row-major input`；调用者可先显式执行 Chapter02 的 `contiguous()`，再提交新的视图。

`try_compute_tensor_views_into()` 先在临时连续结果中计算，后端成功后才写入调用方的输出视图。这个提交步骤用于保证形状、存储或后端执行失败时已有输出保持不变。`try_compute_tensor_views()` 继续返回 `Matrix`，但它只是兼容教学入口，不再是跨章输出类型。

## 算子、计算后端与内核

<a href="../_code/chapter03_matmul/src/matmul_op.cpp"><code>matmul_op.cpp</code></a> 将计算拆成三层：

| 层 | 决定什么 | 当前对象 |
| --- | --- | --- |
| 算子 | 输入输出和数学语义 | `MatMulOp` |
| 计算后端 | 当前主机请求使用哪个实现 | `CpuReferenceMatmulBackend` |
| 内核/计算库 | 怎样完成循环或设备库调用 | CPU 三重循环；独立 CUDA 目标使用 `cublasSgemm` |

CPU 与 CUDA 都保持 `A × B` 的数学语义，但指针所在内存不同。行优先到 cuBLAS 列优先的转换、设备内存分配、CUDA 流和句柄生命周期位于 `code/cuda`，不能把 `Matrix::data` 的主机指针直接传给 cuBLAS。

## 验证与衔接

运行 `bash code/chapter03_matmul/scripts/run.sh`，确认合法矩阵、Chapter02 连续视图和显式物化结果都得到正确输出；形状不匹配、空引用、错误数据类型/设备、越界存储和未物化的非连续布局返回明确错误，并且不覆盖已有输出。[`CPU 参考实现`](./03_cpu_reference.md) 将建立可读的数值基线。
