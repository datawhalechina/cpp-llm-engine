# Chapter 07 代码：分页 KV Cache

本章把已经生成的 K/V 从连续数组迁移到固定大小的物理块。请求只保存逻辑词元到物理块的映射，Attention 根据映射读取已提交数据。先阅读 [`Chapter07 导读`](../../source/chapter07_kv_cache/next/intro.md)，再运行：

```bash
bash code/chapter07_kv_cache/scripts/run.sh
```

## 资产总览

| 模块 | 学习内容 | 验证 |
| --- | --- | --- |
| `BlockTable` | 把请求内的词元位置映射为物理块和块内位置 | `chapter07_block_pool_test` |
| `KvBlockPool` | 分配、提交、裁剪和释放固定大小物理块 | `chapter07_block_pool_test` |
| `CpuPagedKvCache` | 按层保存 K/V 数值，并从分页数据执行 CPU Attention | `chapter07_paged_kv_cache_test` |
| `run_real_paged_kv.sh` | 比较 TinyLlama 连续缓存与分页缓存的真实结果 | `real_tinyllama_paged_kv_test` |

一次写入分为三个步骤：先为请求预留容量，再写入每一层的 K/V，最后提交新的词元长度。提交前读取不到新位置；批量提交会先验证所有请求，任一请求不完整时不修改其他请求。

## 阅读顺序

1. 从 `block_table.h/.cpp` 理解逻辑位置、物理块和已提交长度。
2. 阅读 `paged_kv_cache.cpp` 的 `write_token()` 与 `commit_tokens()`，观察数据写入和可见性如何分离。
3. 阅读 `attention()`，对照测试中的连续数组参考实现。
4. 阅读 `trim_to_tokens()` 与 `release()`，确认失败回滚和块复用前会清理旧数据。

Chapter06 产生词元并推进请求状态，Chapter07 保存模型执行产生的 K/V，Chapter08 决定哪些请求进入下一轮执行。缓存不选择请求，调度器也不直接管理 K/V 数值。

## 验证

默认脚本构建本章并运行容量、事务、跨块等价和复用隔离测试：

```bash
bash code/chapter07_kv_cache/scripts/run.sh
```

已准备模型和 llama.cpp CPU 构建时，运行真实模型验收：

```bash
bash code/chapter07_kv_cache/scripts/run_real_paged_kv.sh
```

## 扩展位置

`CpuPagedKvCache` 是可核对数值的 CPU 参考实现。后续可以在保持页表、提交和回收语义不变的前提下替换 GPU 分页 Attention，或在块池之上增加前缀复用；多请求选择仍由 Chapter08 负责。
