# Chapter 02：Tensor Ops Tensor 视图与共享存储

本目录实现 CPU Float32 的 1-4 维 Tensor、共享 Storage View、轴切片与轴交换，以及将非连续 View 显式物化的 `contiguous()`。完整学习说明见 [`source/chapter02_tensor_ops`](../../source/chapter02_tensor_ops/intro.md)。

## 本章资产总览

| 资产 | 作用 |
| --- | --- |
| `CMakeLists.txt` | library、demo 和 test target |
| `src/tensor.cpp` | reshape、slice、transpose、连续化和只读/可写 `TensorViewRef` 导出 |
| `examples/main_tensor_ops.cpp` | 最小 view 操作演示 |
| `tests/chapter02_tensor_ops_test.cpp` | shape、offset、别名、连续化和边界测试 |
| `tests/chapter02_nd_tensor_test.cpp` | 1D、3D、4D 索引与 View 测试 |
| `scripts/run.sh` | 隔离构建、CTest 和 demo 入口 |

## 运行与验证

从仓库根目录执行：

```bash
bash code/chapter02_tensor_ops/scripts/run.sh
```

重点观察：N 维索引是否按 shape/stride 落到正确元素，轴交换是否共享 Storage，非连续 View 调用 `contiguous()` 后是否获得独立的连续 Storage。
