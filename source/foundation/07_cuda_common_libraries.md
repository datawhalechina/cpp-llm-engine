# F07 CUDA 常用库

CUDA 库按问题选择，不按名气堆叠。Foundation 先建立问题与工具的映射，具体计算核位于 `code/cuda`，算子接口、调度和性能证据分别由 Chapter03–05、08、09 展开。

## 计算问题与 CUDA 库

| 问题 | 常见工具 | 教程中的位置 |
| --- | --- | --- |
| 稠密矩阵乘法 | cuBLAS | Chapter03 后端 |
| 计算核模板和分块 | CUTLASS | Chapter03/05 扩展阅读 |
| 设备端容器和排序 | Thrust、CUB | 算子或辅助路径 |
| 重复启动开销 | CUDA Graph | Chapter08 |
| 多 GPU 通信 | NCCL | 服务扩展，不是 CPU 主线 |
| 深度学习算子集合 | cuDNN | 外部框架对照 |

## 选择顺序

```text
CPU 参考实现
  -> CUDA 计算核 / cuBLAS
  -> 正确性对照
  -> 性能基准
  -> 再考虑 Graph、异步拷贝或通信
```

异步拷贝和 CUDA Graph 是建立同步基线后的优化手段。单 GPU 主线先完成设备执行与正确性对照，再把 NCCL 留给多 GPU 扩展。

## 最小示例

```cpp
// CPU 参考实现和 CUDA 计算核使用相同的输入、输出和边界条件。
__global__ void add_kernel(const float* a, const float* b, float* c, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) c[i] = a[i] + b[i];
}
```

FQ10 用这个最小问题建立 CPU/CUDA 对照；cuBLAS、CUDA Graph、CUB 和 NCCL 分别在对应扩展主题中继续学习。

## 验证与练习

验证顺序是：解释库解决的问题、指出它位于算子还是运行时、用 CPU 结果对照 GPU 结果。对应练习见 [Foundation 正文与练习映射](./exercise_map.md)，完成 FQ10；继续实践时进入 `code/cuda` 的设备内存、CUDA 流、计算核和 cuBLAS 测试。
