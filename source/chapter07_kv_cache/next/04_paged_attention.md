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

三阶段实现先把每个逻辑 key 的分数写入完整 score workspace，再执行 Softmax 和 Value 聚合；它保留清晰的数值基线。分页融合候选把页表映射放进在线 Softmax 循环，每次读取 K/V 前计算 `physical_block + slot`，只保留运行最大值、归一化分母和输出累加，不物化完整 score。两条路径使用同一 BlockTable 和因果可见范围，差别只在中间结果是否写回显存。

EXP-GPU-07 的真实 TinyLlama A/B 中，两轮 Prefill 和 Decode 中位数方向不一致，因此三阶段仍是默认路径。Nsight 附件显示融合 Attention 的 kernel 合计时间比三阶段约少 `32.3%`，但 Q4_K/Q6_K Linear 占 GPU kernel 时间约 `99.2%`。融合实现用于观察在线状态与分页寻址怎样组合，也说明局部 kernel 改善只有在端到端占比足够高时才会成为用户可见收益。

| 输入 | 作用 |
| --- | --- |
| query | 一个或多个位置的查询头向量 |
| 请求 ID | 选择请求的 `BlockTable` |
| 层号 | 选择当前解码层的 K/V |
| 已提交词元数 | 限制可读取的历史范围 |
| 物理数据区 | 提供实际 key/value 行 |

## Prefill 为什么需要多个 Query

Decode 每次只有一个 Query，位置就是已提交词元数。Prefill 会一次处理一个连续 chunk；假设缓存已有 `P` 个词元，本次输入 `T` 个词元，那么 Query `i` 的绝对位置是 `P + i`，最多读取 `P + i + 1` 行 K/V。

| 输入量 | 含义 |
| --- | --- |
| `query_position_offset = P` | 当前 chunk 第一个 Query 的绝对位置 |
| `query_tokens = T` | 当前 chunk 中的 Query 数量 |
| `key_tokens = P + T` | 已提交历史与当前 chunk 写入的 K/V 总数 |
| `visible_keys(i) = P + i + 1` | 因果掩码为 Query `i` 保留的可见范围 |

实现先批量计算当前 chunk 的 Q/K/V，再把每个 K/V 行按 BlockTable 写入物理槽位。Attention 随后读取“历史 + 当前 chunk”，因果掩码阻止前面的 Query 看到 chunk 中更靠后的 K/V。分页路径不需要把离散物理块拼回连续 Tensor。

控制面仍保持事务边界：预留只提供写入位置，设备 payload 写入和所有层计算成功后才把已提交词元数推进到 `P + T`；任何一层失败都裁剪回 `P`。

## 怎样证明寻址没有改变结果

测试使用同一组 query、key 和 value 计算两次：连续参考按 `[token, kv_head, head_dimension]` 读取；分页实现让多个 Query 和 K/V 跨越非连续物理块。两组输出逐元素接近，才能说明分页只改变地址，没有改变 Attention 语义。测试还要让 chunk 跨过块边界，并检查 Query `i` 看不到未来位置。

`CpuPagedKvCache::attention()` 是便于阅读和核对的 CPU 参考实现。它逐词元查询页表并使用临时分数数组，不代表生产级融合内核；设备实现可以改变并行方式和数据布局，但必须保持相同输入契约与数值基线。

## 验证连续与分页等价

通过条件包括多 Query 因果遮罩、跨块读取、GQA 头映射、连续与分页结果一致，以及物理块释放复用后无请求串扰。CUDA 专项测试还会把分页 Prefill 与逐词元 Decoder 对照。

```bash
cmake -S code/chapter07_kv_cache -B /tmp/cpp-llm-chapter07-attention \
  -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/cpp-llm-chapter07-attention \
  --target chapter07_paged_kv_cache_test -j2
ctest --test-dir /tmp/cpp-llm-chapter07-attention \
  -R '^chapter07_paged_kv_cache_test$' --output-on-failure
```
