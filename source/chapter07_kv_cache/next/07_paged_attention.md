# 07.7 让 Attention 按页表读取历史 K/V

分页 Attention 与连续 Attention 使用相同的缩放点积、Softmax 和加权求和。两者的差别在 K/V 地址：连续实现用词元位置直接计算偏移，分页实现先经过 BlockTable 找到物理块和槽位。

## 寻址发生在哪里

对每个 query head 和历史词元位置 `t`，分页路径执行：

```text
t
  -> logical_block + slot
  -> BlockTable[logical_block]
  -> physical_block + slot
  -> key/value row
  -> score and weighted sum
```

GQA 仍用 `query_head / queries_per_kv_head` 选择共享的 KV head。页表不参与数值公式，只替换 K/V 行的地址计算。

| 输入 | 作用 |
| --- | --- |
| query | 当前词元各查询头的向量 |
| request ID | 选择请求对应的 BlockTable |
| layer | 选择当前解码层的 K/V |
| committed token count | 限制可读取的历史范围 |
| physical payload | 提供实际 key/value 行 |

## 怎样证明分页没有改变结果

测试为同一组 query、key 和 value 计算两次结果：连续参考按 `[token, kv_head, head_dimension]` 读取，分页实现让三个词元跨越两个物理块。两组输出逐元素接近，才能说明寻址变化没有改变 Attention 语义。

当前 `CpuPagedKvCache::attention()` 是便于阅读和核对的 CPU 参考实现，不是面向吞吐优化的融合内核。它逐词元查询页表并使用临时分数数组；后续设备实现可以改变并行方式，但必须保持相同输入契约和数值基线。

## 验证连续与分页结果一致

```bash
cmake -S code/chapter07_kv_cache -B /tmp/cpp-llm-chapter07-attention \
  -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/cpp-llm-chapter07-attention \
  --target chapter07_paged_kv_cache_test -j2
ctest --test-dir /tmp/cpp-llm-chapter07-attention \
  -R '^chapter07_paged_kv_cache_test$' --output-on-failure
```

通过条件包括跨块读取、GQA 头映射、连续/分页输出一致和释放复用无串扰。
