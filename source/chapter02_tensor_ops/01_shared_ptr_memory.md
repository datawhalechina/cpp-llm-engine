# 02.1 共享底层存储

Chapter01 的权重张量已经区分逻辑形状和物理字节。Chapter02 使用独立的 Float32 教学张量，把这个概念扩展成可操作的视图：多个视图共享同一份底层存储，形状或窗口变化不必复制数据。

## 底层存储与视图怎样分开

底层存储（storage）保存元素，视图（view）保存观察方式：

| 视图字段 | 回答的问题 |
| --- | --- |
| `shape` | 每个维度有多长 |
| `stride` | 下标增加 1 时跨过多少个元素 |
| `offset` | 当前视图从底层存储的哪个元素开始 |
| 存储句柄 | 哪个对象保证底层数据仍然存活 |

维数（`rank`）是形状中轴的数量。教程实现接受 1–4 维，连续布局的步长从最后一维向前计算：

| 维数 | 示例形状 | 连续步长 | 推理对象 |
| --- | --- | --- | --- |
| 1 | `[hidden]` | `[1]` | 归一化权重、单个向量 |
| 2 | `[tokens, hidden]` | `[hidden, 1]` | 隐藏状态、权重矩阵 |
| 3 | `[tokens, heads, head_dim]` | `[heads×head_dim, head_dim, 1]` | Q/K/V |
| 4 | `[batch, tokens, heads, head_dim]` | `[tokens×heads×head_dim, heads×head_dim, head_dim, 1]` | 注意力、KV Cache 执行输入 |

<a href="../_code/chapter02_tensor_ops/include/chapter02_tensor_ops/tensor.h"><code>tensor.h</code></a> 使用 `shared_ptr<vector<float>>` 保存共享所有权。复制张量只复制视图描述和存储句柄，不复制元素：

```cpp
Tensor original(2, 2, 1.0f);
Tensor view = original;
```

只要任一视图仍持有底层存储，数据就不会提前释放。`shares_storage_with()` 和 `storage_use_count()` 让这种关系可以被测试观察。

## 共享存储怎样支持后续操作

`reshape` 复用整个观察窗口，只改变形状和连续步长；`slice` 复用某个轴的局部窗口；`transpose` 交换两个轴的形状和步长。三种操作都建立在同一个共享存储上。

注意力的局部窗口和 KV 数据也需要类似的“数据与观察方式分离”，但缓存分配和淘汰由 Chapter06/09 处理；本章只验证视图的地址与生命周期。

## 验证与衔接

```bash
bash code/chapter02_tensor_ops/scripts/run.sh
```

测试应确认父张量与视图共享底层存储，任一对象销毁都不会造成提前释放或重复释放，修改共享元素后其他视图能观察到相同结果。接下来用 [`reshape`](./02_reshape.md) 改变整份数据的形状解释。
