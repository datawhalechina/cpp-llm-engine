# Chapter 02：张量视图与共享存储

本目录实现 CPU Float32 的 1–4 维张量。`reshape()`、`slice()` 和 `transpose()` 改变形状、步长或偏移，并继续共享原存储；`contiguous()` 在布局不连续时复制数据。原理和图解见 [Chapter02 导读](../../source/chapter02_tensor_ops/intro.md)。

## 代码架构

```text
Tensor：shared_ptr<vector<float>> + shape + strides + offset
  ├─ reshape / slice / transpose ──> 共享 Storage 的 Tensor View
  ├─ contiguous ───────────────────> 连续时共享；非连续时复制为独立 Tensor
  └─ view_ref / mutable_view_ref ──> Core 非拥有引用 ──> Chapter03–05 算子
```

`Tensor` 管理存储生命周期，因此它产生的另一个 `Tensor` 视图可以在原对象销毁后继续使用。`TensorViewRef` 借用地址且不增加引用计数，调用方需要保证对应 `Tensor` 在引用使用期间仍然存活。

`try_reshape()`、`try_slice()`、`try_transpose()` 和 `try_contiguous()` 先检查输入，再提交输出；输出与源对象相同时，失败会保留原视图。跨章引用携带形状、步长、偏移、数据类型、设备和存储字节数。后续算子检查布局，需要连续输入时由调用方显式执行 `contiguous()`。

### 操作与存储行为

| 操作 | 是否共享存储 | 前置条件 | 结果布局 |
| --- | --- | --- | --- |
| `reshape` | 是 | 输入连续且元素总数不变 | 按新形状生成连续步长，保留原偏移 |
| `slice` | 是 | 轴、起点和长度位于原形状内 | 保留原步长，调整形状与偏移 |
| `transpose` | 是 | 两个轴有效 | 交换对应形状和步长，通常变为非连续 |
| `contiguous` | 连续输入共享；非连续输入复制 | 输入对象有效 | 连续行主序布局 |
| `view_ref` | 非拥有引用 | 拥有存储的 `Tensor` 在使用期间存活 | 保留形状、步长、偏移、类型和设备 |

本章的拥有对象使用 CPU Float32。设备张量由 `CudaTensor` 独立管理；新增数据类型或设备时，通过相同的核心视图结构向算子提供地址和布局。

## 阅读顺序与资产

| 资产 | 作用 |
| --- | --- |
| `include/chapter02_tensor_ops/tensor.h` | 先看公开操作和返回方式 |
| `src/tensor.cpp` | 再看地址计算、视图变换和连续化 |
| `examples/main_tensor_ops.cpp` | 观察形状、步长、偏移和共享关系 |
| `tests/chapter02_tensor_ops_test.cpp` | 核对二维视图、生命周期和失败行为 |
| `tests/chapter02_nd_tensor_test.cpp` | 核对 1D、3D、4D 索引与轴变换 |

阅读实现时先看 `checked_storage_index()` 如何把 N 维索引转换为存储位置，再看三个 `try_*` 视图操作，最后看 `try_contiguous()` 如何按逻辑索引复制非连续数据。

## 运行与验证

从仓库根目录执行：

```bash
bash code/chapter02_tensor_ops/scripts/run.sh
```

脚本先输出二维视图和三维 Q/K/V 的布局，再运行两个 CTest。重点核对 N 维索引是否按形状、步长和偏移找到元素，视图修改是否反映到原张量，以及连续化后写入是否相互隔离。

## 扩展入口

Chapter03–05 通过核心 `TensorViewRef` 接收本章数据。后续增加设备内存或新数据类型时，可以实现新的拥有对象并继续导出同一种核心引用。
