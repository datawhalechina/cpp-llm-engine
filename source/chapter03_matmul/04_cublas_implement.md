# 03.4 接入 cuBLAS 计算后端

CPU 内核按行优先读取矩阵，cuBLAS 默认按列优先解释内存。本节跟踪 `CublasMatmulBackend` 怎样交换输入矩阵和维度参数，在不复制数据的情况下得到相同输出。实现和测试分别位于 <a href="../_code/cuda/src/cublas_matmul_backend.cpp"><code>cublas_matmul_backend.cpp</code></a> 与 <a href="../_code/cuda/tests/cuda_cublas_matmul_test.cu"><code>cuda_cublas_matmul_test.cu</code></a>。

## 行优先布局映射

需要计算的矩阵乘法是：

```text
C[M,N] = A[M,K] * B[K,N]
```

cuBLAS 默认按列优先布局解释内存。当前实现不复制或转置数据，而是交换传入的 A/B：`B[K,N]` 的行优先内存被解释为列优先 `B^T[N,K]`，`A[M,K]` 同理被解释为 `A^T[K,M]`。因此 cuBLAS 实际计算：

```text
C^T[N,M] = B^T[N,K] * A^T[K,M]
```

| SGEMM 参数 | 当前值 | 原因 |
| --- | --- | --- |
| 第一个矩阵 | `request.rhs` | 作为 `B^T[N,K]` 解释 |
| 第二个矩阵 | `request.lhs` | 作为 `A^T[K,M]` 解释 |
| `transa/transb` | `CUBLAS_OP_N / CUBLAS_OP_N` | 内存视角已经完成转置解释 |
| `m/n/k` | `N / M / K` | 输出按 `C^T[N,M]` 计算 |
| `lda/ldb/ldc` | `N / K / N` | 对应三个列优先矩阵的行数 |

## Stream、Handle 与错误状态

`CublasMatmulBackend` 持有一个可以移动的 `cublasHandle_t`。每次 `compute()` 接收 `CudaStream` 并调用 `cublasSetStream`，本次 SGEMM 因而进入指定 Stream。

`CublasStatus` 保留原始 `cublasStatus_t`、操作名和可读错误，并能转换为核心层的 `Status`。计算后端在调用库函数前拒绝空句柄、无效 CUDA 流、空指针、零维和超出 cuBLAS `int` 范围的维度。

## 运行 GPU 对照

从仓库根目录运行：

```bash
CUDA_HOME=/usr/local/cuda-12.8 bash code/cuda/scripts/run.sh
```

测试用 `1x1x1`、`2x3x4`、`7x5x9` 和 `31x33x17` 与 CPU 三重循环逐元素对照。固定 `256x256x256` 工作负载经过 1 次预热后使用 CUDA 事件记录 10 轮 SGEMM；设备到主机的数据复制和结果检查位于计时区间之外。
