# Chapter02 N 维 Tensor 重构规格

Chapter02 当前使用 `rows / cols / row_stride / col_stride` 保存二维布局，公共 `TensorContract` 已使用 `shape + strides` 描述任意 rank。下一轮重构让教学 Tensor 与公共契约使用同一种表示，同时保留 Chapter03 已使用的二维接口。

## 目标数据模型

第一阶段支持 CPU Float32、1–4 维非空 Tensor。Storage 继续由 `shared_ptr<vector<float>>` 持有，View 只保存以下元数据：

| 字段 | 类型 | 约束 |
| --- | --- | --- |
| `shape` | `vector<size_t>` | rank 为 1–4，各维大于 0 |
| `strides` | `vector<size_t>` | 数量与 rank 相同，单位为元素 |
| `offset` | `size_t` | 当前 View 首个逻辑元素在 Storage 中的位置 |
| `storage` | `shared_ptr<vector<float>>` | 多个 View 可以共享所有权 |

逻辑索引到 Storage 的地址使用统一公式：

```text
storage_index = offset + sum(indices[axis] * strides[axis])
```

连续布局从最后一维向前计算 stride。长度为 1 的维度不影响连续性判断，因为该轴只有下标 0。

## 接口与二维兼容

N 维接口负责表达真实模型布局；现有二维接口作为便捷包装保留，避免 Chapter03 在同一轮重写。

| 类型 | 接口 | 行为 |
| --- | --- | --- |
| 构造 | `Tensor(shape, value)` | 创建连续 1–4 维 Tensor |
| 查询 | `rank()`、`shape()`、`strides()`、`numel()` | 返回布局元数据 |
| 索引 | `at(indices)` | 按 N 维索引访问元素并检查范围 |
| 重塑 | `reshape(shape)` | 只允许连续 View，元素数不变 |
| 切片 | `slice(axis, start, count)` | 改变指定轴的 shape 和 offset，不复制 |
| 交换轴 | `transpose(axis_a, axis_b)` | 交换两个轴的 shape/stride，不复制 |
| 连续化 | `contiguous()` | 连续时复用 Storage，非连续时按逻辑顺序复制 |
| 二维包装 | `Tensor(rows, cols)`、`at(row,col)`、`rows()`、`cols()` | 要求 rank=2，转调 N 维实现 |

失败接口继续返回 `Status`，便捷接口失败时返回无效 Tensor。`rows()`、`cols()` 和二维 `at()` 在非二维 Tensor 上不能猜测维度含义，应返回明确失败或抛出边界异常。

`try_reshape`、`try_slice`、`try_transpose` 和 `try_contiguous` 允许输出参数指向源 Tensor。实现必须先基于源 View 完成验证和结果构造，再替换输出：成功时原地更新 View，失败时源 Tensor 保持不变。非别名输出在失败时重置为无效 Tensor。

## 算子 Rank 约束

Tensor 支持 1–4 维不代表每个算子自动支持所有维度。调用方先检查 rank，再解释 shape：

| 消费方 | 接受的 rank | 当前语义 |
| --- | --- | --- |
| Chapter03 MatMul adapter | 2 | `[M,K] × [K,N]` |
| Chapter04 Q/K/V 布局 | 3 | `[tokens, heads, head_dim]` |
| Chapter04 Attention/KV 执行输入 | 4 | `[batch, tokens, heads, head_dim]` |
| 多模态图像输入 | 4 | `[batch, channels, height, width]` |

Chapter03 不负责批量 MatMul，也不把高维输入自动展平。后续批量算子应建立独立 contract，避免在适配器中隐藏 shape 解释。

## 迁移顺序与验证

重构按兼容优先执行，任何一步都不能同时改变 Chapter03 的 MatMul 数学语义。

| 顺序 | 修改 | 必须固定的证据 |
| --- | --- | --- |
| 1 | 用 `shape_ / strides_` 替换二维成员 | 现有 Chapter02、03 测试不变并通过 |
| 2 | 增加 N 维构造、查询和索引 | 1D、3D、4D 固定索引正确，非法 rank/索引失败 |
| 3 | 泛化 reshape、slice、transpose | Storage 共享、offset/stride 和连续性正确 |
| 4 | 泛化 contiguous | 3D/4D 非连续 View 按逻辑顺序复制 |
| 5 | 固定输出别名语义 | 原地成功，失败时源 Tensor 不变 |
| 6 | 更新学习者正文和主图 | 文档只描述已经通过测试的接口 |
| 7 | 接入 Chapter04 | Q/K/V shape 由 3D/4D Tensor 实际承载 |

章内测试验证 N 维地址和 View 语义；Chapter02→03 跨章测试固定二维兼容；Chapter02→04 跨章测试留到 Chapter04 开始消费 3D/4D Tensor 时增加。全局回归继续验证 Chapter00–09 没有因内部表示变化而退化。
