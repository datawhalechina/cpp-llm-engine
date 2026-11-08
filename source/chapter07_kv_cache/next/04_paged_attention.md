# 07.4 让 Attention 按页表读取 K/V

分页 Attention 与连续 Attention 使用相同的缩放点积、Softmax 和加权求和。变化只发生在 K/V 寻址：连续实现用词元位置直接计算偏移，分页实现先经 `BlockTable` 找到物理块和槽位。

## 页表怎样进入 Attention

对每个 query head 和历史位置 `t`，分页路径按以下顺序读取：

```text
t
  -> logical_block + slot
  -> BlockTable[logical_block]
  -> physical_block + slot
  -> key/value row
  -> score and weighted sum
```

GQA 仍用 `query_head / queries_per_kv_head` 选择共享的 KV head。页表不参与 Attention 数值公式，只替换 K/V 行的地址来源。

| 输入 | 作用 |
| --- | --- |
| query | 当前词元各查询头的向量 |
| 请求 ID | 选择请求的 `BlockTable` |
| 层号 | 选择当前解码层的 K/V |
| 已提交词元数 | 限制可读取的历史范围 |
| 物理数据区 | 提供实际 key/value 行 |

## 怎样证明寻址没有改变结果

测试使用同一组 query、key 和 value 计算两次：连续参考按 `[token, kv_head, head_dimension]` 读取；分页实现让三个词元跨越两个物理块。两组输出逐元素接近，才能说明分页只改变地址，没有改变 Attention 语义。

`CpuPagedKvCache::attention()` 是便于阅读和核对的 CPU 参考实现。它逐词元查询页表并使用临时分数数组，不代表生产级融合内核；设备实现可以改变并行方式和数据布局，但必须保持相同输入契约与数值基线。

## 验证连续与分页等价

通过条件包括跨块读取、GQA 头映射、连续与分页结果一致，以及物理块释放复用后无请求串扰。

```bash
cmake -S code/chapter07_kv_cache -B /tmp/cpp-llm-chapter07-attention \
  -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/cpp-llm-chapter07-attention \
  --target chapter07_paged_kv_cache_test -j2
ctest --test-dir /tmp/cpp-llm-chapter07-attention \
  -R '^chapter07_paged_kv_cache_test$' --output-on-failure
```
