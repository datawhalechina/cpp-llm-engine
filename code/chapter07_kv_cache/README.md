# Chapter 07 代码资产：KV Cache 与内存管理

本目录把逻辑词元映射、物理块所有权和真实 K/V 数值从 Generation、Scheduler 中拆出。学习说明见 [`Chapter07 导读`](../../source/chapter07_kv_cache/next/intro.md)。先运行：

```bash
bash code/chapter07_kv_cache/scripts/run.sh
```

## 资产总览

| 资产 | 职责 | 验证 |
| --- | --- | --- |
| `block_table.h/.cpp` | `BlockTable`、`KvBlockPool` 与物理槽位定位 | `chapter07_block_pool_test` |
| `paged_kv_cache.h/.cpp` | 分层分页 K/V、提交检查与 CPU 分页 Attention | `chapter07_paged_kv_cache_test` |
| `tests/` | 容量失败、事务、跨块等价和释放复用 | CTest 标签 `cache`、`contract`、`transaction` |
| `scripts/run_real_paged_kv.sh` | 真实 TinyLlama 连续/分页等价、回滚与释放验收 | `real_tinyllama_paged_kv_test` |

Engine 通过 `RealLlamaKvDataPlane` 把真实 TinyLlama Context 绑定到本模块，Decode Attention 直接读取分页 payload；连续 Context 继续作为数值基线。前缀缓存和 GPU 分页内核属于后续扩展，多请求选择属于 Chapter08。

已准备模型和 llama.cpp CPU 构建时，运行真实模型验收：

```bash
bash code/chapter07_kv_cache/scripts/run_real_paged_kv.sh
```
