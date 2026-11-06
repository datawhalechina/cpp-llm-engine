# 07.0 管理可分页的 K/V 与物理内存

## 本章导读

Chapter06 已经固定单请求的 Prefill、Decode 和停止语义。本章接管模型执行产生的 K/V：先用连续缓存建立数值基线，再把逻辑词元位置映射到固定大小物理块，完成分页读写、提交、回滚和释放。Chapter08 只通过这些资源接口安排多请求，不直接拥有 K/V 数据。

| 学习线 | 本章怎样学习 | 可验证产物 |
| --- | --- | --- |
| A 连续基线 | 从 Attention 形状和连续地址开始，确认序列怎样增长 | 连续 K/V 追加、容量拒绝和请求隔离 |
| B 分页数据面 | 跟踪逻辑位置、BlockTable、物理槽位和分页 Attention | 跨块读写、事务、连续/分页输出一致和安全复用 |

| 本章新术语 | 先这样理解 | 首次解释 |
| --- | --- | --- |
| 块表（`BlockTable`） | 保存一个请求的逻辑块到物理块映射 | [`07.3 Block 与槽位`](./03_block_page_layout.md) |
| KV 块池（`KvBlockPool`） | 分配和回收固定大小物理块 | [`07.4 块表与内存池`](./04_block_table_pool.md) |
| 分页 K/V 数据面 | 按层、物理块和槽位保存真实 key/value | [`07.5 分页 K/V`](./05_paged_payload.md) |
| 分页注意力 | 根据块表直接读取离散 K/V 的 Attention | [`07.7 分页 Attention`](./07_paged_attention.md) |

![Chapter 07 主图：逻辑词元怎样经过块表进入分页 K/V](../../assets/diagrams/chapter07-kv-cache.svg)

## 开始学习

先运行 Chapter07 独立入口：

```bash
bash code/chapter07_kv_cache/scripts/run.sh
```

脚本构建 `chapter07_kv_cache` 并运行块池与分页数据面测试。通过时 CTest 显示 `chapter07_block_pool_test` 和 `chapter07_paged_kv_cache_test` 全部成功，覆盖分配失败、跨块读写、逐层提交、回滚和释放复用。

| 顺序 | 路线 | 页面 | 先看源码 | 读完后应看到什么 |
| --- | --- | --- | --- | --- |
| 1 | A | [`07.1 K/V 形状`](./01_kv_shape.md)、[`07.2 连续缓存`](./02_contiguous_kv.md) | Chapter04 `kv_cache.cpp` | Attention 保存什么，连续地址怎样计算 |
| 2 | A→B | [`07.3 Block 与槽位`](./03_block_page_layout.md)、[`07.4 块表与内存池`](./04_block_table_pool.md) | <a href="../../_code/chapter07_kv_cache/src/block_table.cpp"><code>block_table.cpp</code></a> | 逻辑词元怎样获得物理位置 |
| 3 | B | [`07.5 分页 K/V`](./05_paged_payload.md)、[`07.6 缓存事务`](./06_cache_transaction.md) | <a href="../../_code/chapter07_kv_cache/src/paged_kv_cache.cpp"><code>paged_kv_cache.cpp</code></a> | 预留、逐层写入、提交、回滚和释放怎样配合 |
| 4 | B | [`07.7 分页 Attention`](./07_paged_attention.md) | <a href="../../_code/chapter07_kv_cache/tests/paged_kv_cache_test.cpp"><code>paged_kv_cache_test.cpp</code></a> | 离散物理块为什么仍得到连续基线结果 |
| 5 | 扩展 | [`07.8 前缀复用`](./08_prefix_cache.md)–[`07.10 框架对照`](./10_framework_comparison.md) | Engine 真实模型入口、生态卡 | 真实 Decoder 怎样绑定分页数据面，前缀复用还缺什么 |

## 验证与下一步

完成本章后，K/V 的逻辑形状、物理位置、数据写入和生命周期都有独立契约；Chapter08 可以使用缓存事务组织多请求执行，而不修改分页数据面。

| 验收层级 | 执行入口 | 通过条件 | 本章要求 |
| --- | --- | --- | --- |
| L1 连续基线 | `bash code/chapter04_attention/scripts/run.sh`（其中的 `chapter04_attention_test`） | 追加、容量与 Attention 读取通过 | 必做 |
| L2 CPU 分页 | `bash code/chapter07_kv_cache/scripts/run.sh` | 分配失败无泄漏、事务一致、跨块输出等价、释放复用无串扰 | 必做 |
| L3 真实模型分页 | `bash code/chapter07_kv_cache/scripts/run_real_paged_kv.sh` | TinyLlama 22 层 logits、固定 token、失败回滚、连续路径一致和终态释放 | 有模型时必做 |
