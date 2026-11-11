# 02.2 变形与转置

`reshape` 在不改变元素顺序和底层存储的情况下，为连续视图建立新的形状。它适合调整下一步算子的输入形状，不负责重新排列数据。

## 形状与步长怎样变化

形状（`shape`）记录每个维度的长度；步长（`stride`）记录某一维下标增加 1 时跨过的元素数。连续 `[2,3]` 张量的步长是 `[3,1]`。

`reshape` 必须同时满足：

- 新旧元素总数相同；
- 当前视图连续；
- 底层存储和偏移保持不变；
- 新步长按连续布局重新计算。

带非零偏移的完整连续行仍可变形；跨行 `2x2` 切片保留父张量的行步长，通常不连续，因此必须拒绝变形，而不是暗中复制。

## 实现与失败状态

<a href="https://github.com/datawhalechina/cpp-llm-engine/blob/main/code/chapter02_tensor_ops/src/tensor.cpp"><code>tensor.cpp</code></a> 中的 `try_reshape()` 先验证源张量、连续性、维度和元素数量，再构造共享底层存储的新视图：

```cpp
Tensor output;
auto status = tensor.try_reshape(rows, cols, &output);
```

`Status` 区分源张量未就绪、非连续、元素数变化和溢出。便捷接口 `reshape()` 复用同一条路径，失败时返回无效张量。

输出参数可以指向源张量。`tensor.try_reshape(shape, &tensor)` 会先构造临时结果，成功后再替换原视图；验证失败时原张量保持不变。

## 转置交换逻辑轴

二维便捷接口 `transpose()` 把 `[rows, cols]` 变成 `[cols, rows]`。通用接口 `transpose(axis_a, axis_b)` 可以交换 1–4 维张量中任意两个轴。两者都同时交换对应的形状和步长，底层存储、偏移和数值不移动：

```text
原张量： shape [2,3]  stride [3,1]
转置视图：shape [3,2]  stride [1,3]
```

交换轴后的相邻逻辑元素可能不再物理相邻，因此视图通常不连续。需要交给只接受连续布局的算子时，调用者先执行 `contiguous()`，把逻辑顺序物化为新的连续存储。

## 验证与衔接

运行 `bash code/chapter02_tensor_ops/scripts/run.sh`，确认变形只重解释连续元素，转置只交换形状和步长，两个视图都共享原存储；转置连续化后使用独立存储。[`slice`](./03_slice.md) 将继续改变观察窗口。
