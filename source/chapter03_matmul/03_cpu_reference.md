# 03.3 参考实现（CPU）

CPU 参考实现按矩阵乘法公式直接写成三重循环。代码不追求最快，而是让索引、累加方向和输出位置都能逐行检查。

## 三重循环怎样对应矩阵乘法

对于 `A[M,K] × B[K,N]`，输出元素 `C[i,j]` 沿 K 维累加：

```text
C[i,j] = sum(A[i,k] * B[k,j]), k = 0 ... K-1
```

外层 `i` 选择输出行，中层 `j` 选择输出列，内层 `k` 依次取出需要相乘的两个元素：

```cpp
for (std::size_t i = 0; i < M; ++i) {
    for (std::size_t j = 0; j < N; ++j) {
        float sum = 0.0f;
        for (std::size_t k = 0; k < K; ++k) {
            sum += A[i * K + k] * B[k * N + j];
        }
        C[i * N + j] = sum;
    }
}
```

循环结束时，`sum` 就是左矩阵第 `i` 行与右矩阵第 `j` 列的点积。这个写法直接暴露索引、累加维度和行优先地址，适合排查输入顺序、转置和形状错误。

## 请求怎样进入 CPU 内核

`MatMulOp::try_compute()` 将矩阵适配为 `MatmulRequest`；`try_compute_tensor_views_into()` 接收 Chapter02 的两个只读输入视图和一个可写输出视图；`CpuReferenceMatmulBackend` 负责分派；<a href="../_code/chapter03_matmul/src/cpu_kernel.cpp"><code>cpu_kernel.cpp</code></a> 校验指针和尺寸后执行循环。返回 `Matrix` 的 `try_compute_tensor_views()` 继续作为教学兼容入口。

SIMD、分块和 GPU 可以换一种更快的计算方法，但输出仍要与这段循环比较。

## 运行并观察

运行 `bash code/chapter03_matmul/scripts/run.sh`。固定示例应得到 `value(0,0)=58` 和 `value(1,1)=154`；跨章测试还会检查非零偏移视图、切片连续化和转置连续化。具备 CUDA 环境时，可以进入 [`03.4`](./04_cublas_implement.md) 比较 cuBLAS 与这段循环的结果。
