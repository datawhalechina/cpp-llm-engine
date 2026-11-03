# 03.4 接入 cuBLAS 计算后端

本节用已验证的 `cublasSgemm` 计算后端说明行优先与列优先布局的适配。CPU 参考实现和 CUDA 后端复用 `MatmulRequest` 的形状与数学语义；CUDA 请求中的三个指针必须指向设备内存，并与显式 CUDA 流一起通过独立 GPU 目标执行。

项目的矩阵乘法语义不随计算后端改变。实现和测试入口分别位于 <a href="../_code/cuda/src/cublas_matmul_backend.cpp"><code>cublas_matmul_backend.cpp</code></a> 与 <a href="../_code/cuda/tests/cuda_cublas_matmul_test.cu"><code>cuda_cublas_matmul_test.cu</code></a>。

## 行优先布局映射

核心层固定的数学语义是：

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

## 资源与错误边界

`CublasMatmulBackend` 独占只能移动的 `cublasHandle_t`，不使用进程全局句柄。每次 `compute()` 显式接收 `CudaStream` 并调用 `cublasSetStream`，因此句柄与执行上下文可以在下一阶段一起管理。

`CublasStatus` 保留原始 `cublasStatus_t`、操作名和可读错误，并能转换为核心层的 `Status`。计算后端在调用库函数前拒绝空句柄、无效 CUDA 流、空指针、零维和超出 cuBLAS `int` 范围的维度。

## 验证与衔接

从仓库根目录运行：

```bash
CUDA_HOME=/usr/local/cuda-12.8 bash code/cuda/scripts/run.sh
```

测试用 `1x1x1`、`2x3x4`、`7x5x9` 和 `31x33x17` 与 CPU 三重循环逐元素对照。固定 `256x256x256` 工作负载经过 1 次预热后使用 CUDA 事件记录 10 轮 SGEMM；设备到主机的数据复制和结果检查位于计时区间之外。
