# 03.3 参考实现（CPU）

高性能计算后端需要一个容易检查的数值基线。CPU 参考实现使用与数学定义一致的三重循环，不承担最终性能目标。

## 三重循环怎样对应矩阵乘法

对于 `A[M,K] × B[K,N]`，输出元素 `C[i,j]` 沿 K 维累加：

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

该实现直接暴露索引、累加维度和行优先地址，适合排查输入顺序、转置和形状错误。

## 请求怎样进入 CPU 内核

`MatMulOp::try_compute()` 将矩阵适配为 `MatmulRequest`；`try_compute_tensor_views_into()` 接收 Chapter02 的两个只读输入视图和一个可写输出视图；`CpuReferenceMatmulBackend` 负责分派；<a href="../_code/chapter03_matmul/src/cpu_kernel.cpp"><code>cpu_kernel.cpp</code></a> 校验指针和尺寸后执行循环。返回 `Matrix` 的 `try_compute_tensor_views()` 继续作为教学兼容入口。

参考实现保持朴素，后续 SIMD、分块或 GPU 优化必须继续与它比较。优化路径改变执行成本，不能改变矩阵乘法语义。

## 验证与衔接

运行 `bash code/chapter03_matmul/scripts/run.sh`，固定示例应得到 `value(0,0)=58` 和 `value(1,1)=154`。跨章测试还会验证非零偏移视图得到 `32`，切片连续化后得到 `[5,11]`，转置连续化后得到 `[9,12]`。可选的 [`cuBLAS 计算后端`](./04_cublas_implement.md) 使用同一组 `MatmulRequest` 形状字段和数值基线，但指针指向独立分配的设备内存。
