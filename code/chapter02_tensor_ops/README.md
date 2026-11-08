# Chapter 02：Tensor 视图与共享存储

本目录实现 CPU Float32 的 1–4 维 Tensor。`reshape()`、`slice()` 和 `transpose()` 只改变 shape、stride 或 offset，并继续共享原 Storage；`contiguous()` 在布局不连续时复制数据。原理和图解见 [Chapter02 导读](../../source/chapter02_tensor_ops/intro.md)。

## 代码架构

```text
Tensor：shared_ptr<vector<float>> + shape + strides + offset
  ├─ reshape / slice / transpose ──> 共享 Storage 的 Tensor View
  ├─ contiguous ───────────────────> 独立、连续的 Tensor
  └─ view_ref / mutable_view_ref ──> Core 非拥有引用 ──> Chapter03–05 算子
```

`Tensor` 负责 Storage 生命周期，所以由它产生的另一个 `Tensor` View 可以在原对象销毁后继续使用。`TensorViewRef` 只借用地址，不增加引用计数，调用方必须保证对应 `Tensor` 仍然存活。

`try_reshape()`、`try_slice()`、`try_transpose()` 和 `try_contiguous()` 先检查输入，再提交输出；输出与源对象相同时，失败不会改变原视图。跨章引用携带 shape、stride、offset、dtype、device 和存储字节数，后续算子负责决定是否接受该布局，不会自动连续化。

## 阅读顺序与资产

| 资产 | 作用 |
| --- | --- |
| `include/chapter02_tensor_ops/tensor.h` | 先看公开操作和返回方式 |
| `src/tensor.cpp` | 再看地址计算、View 变换和连续化 |
| `examples/main_tensor_ops.cpp` | 观察 shape、stride、offset 和共享关系 |
| `tests/chapter02_tensor_ops_test.cpp` | 核对二维 View、生命周期和失败行为 |
| `tests/chapter02_nd_tensor_test.cpp` | 核对 1D、3D、4D 索引与轴变换 |

阅读实现时先看 `checked_storage_index()` 如何把 N 维索引转换为 Storage 位置，再看三个 `try_*` View 操作，最后看 `try_contiguous()` 如何按逻辑索引复制非连续数据。

## 运行与验证

从仓库根目录执行：

```bash
bash code/chapter02_tensor_ops/scripts/run.sh
```

脚本先输出二维视图和三维 Q/K/V 的可观察布局，再运行两个 CTest。重点核对 N 维索引是否按 shape、stride 和 offset 找到元素，View 修改是否反映到原 Tensor，以及连续化后写入是否相互隔离。

## 扩展入口

Chapter03–05 通过 Core `TensorViewRef` 接收本章数据，不需要依赖 `Tensor` 类。后续增加设备内存或新 dtype 时，应实现新的拥有对象并继续导出 Core 引用，而不是在本章 CPU Float32 教学类中加入设备分支。
