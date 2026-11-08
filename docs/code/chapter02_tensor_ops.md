# Chapter 02：Tensor 视图与共享存储

本章代码用 CPU Float32 Tensor 演示共享 Storage、视图变换和显式连续化。原理与学习顺序见 [Chapter02 导读](../chapter02_tensor_ops/intro.md)。

## 代码架构

```text
Tensor：共享 Storage + shape + strides + offset
  ├─ reshape / slice / transpose ──> 共享 Storage
  ├─ contiguous ───────────────────> 必要时复制到独立连续存储
  └─ view_ref / mutable_view_ref ──> Core 非拥有视图 ──> Chapter03–05
```

`Tensor` 持有 Storage，公共视图只借用地址。`try_*` 在检查成功后提交结果，原地调用失败时保留原视图；后续算子不会自动连续化非连续输入。

## 阅读顺序与资产

| 文件 | 作用 |
| --- | --- |
| <a href="/_code/chapter02_tensor_ops/include/chapter02_tensor_ops/tensor.h"><code>tensor.h</code></a> | 公开 View 操作、Storage 所有权和 Core 引用接口 |
| <a href="/_code/chapter02_tensor_ops/src/tensor.cpp"><code>src/tensor.cpp</code></a> | N 维地址计算、View 变换、连续化与失败提交 |
| <a href="/_code/chapter02_tensor_ops/examples/main_tensor_ops.cpp"><code>examples/main_tensor_ops.cpp</code></a> | 二维视图和三维 Q/K/V 布局示例 |
| <a href="/_code/chapter02_tensor_ops/tests/chapter02_tensor_ops_test.cpp"><code>tests/chapter02_tensor_ops_test.cpp</code></a> | 二维兼容接口、生命周期和失败路径 |
| <a href="/_code/chapter02_tensor_ops/tests/chapter02_nd_tensor_test.cpp"><code>tests/chapter02_nd_tensor_test.cpp</code></a> | 1D、3D、4D 与原地失败原子性 |

## 运行与验证

```bash
bash code/chapter02_tensor_ops/scripts/run.sh
```

脚本先输出示例，再执行两个 CTest。测试覆盖 shape、stride、offset、共享关系、连续化、边界、生命周期以及只读/可写公共视图。

## 扩展入口

Chapter03–05 通过 Core `TensorViewRef` 使用本章数据。CUDA 与多模态拥有各自存储对象，只复用布局契约，不从本章 Tensor 隐式迁移。
