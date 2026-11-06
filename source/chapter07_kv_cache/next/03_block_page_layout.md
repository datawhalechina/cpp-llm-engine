# 07.3 把逻辑词元映射到 Block 和槽位

分页缓存把固定数量的词元组成一个 Block。请求仍看到从 0 开始的连续逻辑位置，缓存管理器把每个逻辑 Block 映射到一个物理 Block ID；物理 ID 不要求连续。

## 两步地址转换

设 `block_size = 3`，逻辑词元位置 `token_index` 先转换为：

```text
logical_block = token_index / block_size
slot          = token_index % block_size
```

BlockTable 再把 `logical_block` 查成 `physical_block`。最终位置由 `{physical_block, slot}` 表示。

| 逻辑词元 | 逻辑 Block | Slot | 示例物理 Block |
| --- | --- | --- | --- |
| 0 | 0 | 0 | 2 |
| 1 | 0 | 1 | 2 |
| 2 | 0 | 2 | 2 |
| 3 | 1 | 0 | 0 |
| 4 | 1 | 1 | 0 |
| 5 | 1 | 2 | 0 |

请求的逻辑序列仍是 `[0,1,2,3,4,5]`，物理 payload 可以分散在 Block 2 和 Block 0。Attention 按 BlockTable 逐位置读取，不需要先把两块拼成连续数组。

## Payload 怎样增加层和 Head

`CpuPagedKvCache` 当前使用以下物理布局：

```text
[layer][physical_block][slot][kv_head][head_dimension]
```

完整元素偏移等价于：

```text
((((layer * total_blocks + physical_block) * block_size + slot)
   * Hkv + kv_head) * Dh + dimension)
```

控制面只保存物理 Block ID、Slot 和有效 token 数；数据面保存真正的 key/value 浮点数。源码分别位于 <a href="/_code/chapter07_kv_cache/include/chapter07_kv_cache/block_table.h"><code>block_table.h</code></a> 和 <a href="/_code/chapter07_kv_cache/src/paged_kv_cache.cpp"><code>paged_kv_cache.cpp</code></a>。

## 验证跨 Block 读写

```bash
cmake -S code/chapter07_kv_cache -B /tmp/cpp-llm-chapter07-paged \
  -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/cpp-llm-chapter07-paged \
  --target chapter07_paged_kv_cache_test -j2
ctest --test-dir /tmp/cpp-llm-chapter07-paged \
  -R '^chapter07_paged_kv_cache_test$' --output-on-failure
```

测试让三个词元跨越两个物理 Block，逐层写入 K/V，再比较分页 Attention 与连续 reference 的输出。下一篇进入 BlockTable 和 KvBlockPool 的分配、容量与所有权。
