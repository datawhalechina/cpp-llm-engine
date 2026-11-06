# 07.4 用 BlockTable 和内存池管理物理块

分页缓存需要同时回答两个问题：一个请求占用了哪些物理块，这些物理块还能容纳多少词元。`BlockTable` 保存单个请求的映射，`KvBlockPool` 保存全局物理块的空闲状态，并按请求 ID 持有对应页表。

## 两个对象分别保存什么

| 对象 | 保存的信息 | 不保存的信息 |
| --- | --- | --- |
| `BlockTable` | 块大小、已提交词元数、物理块 ID 列表 | K/V 浮点数、请求优先级 |
| `KvBlockPool` | 全部物理块的空闲状态、请求到页表的映射 | 每层 K/V、调度队列 |
| `BlockLocation` | 一个物理块 ID 和块内槽位 | 层号、注意力头和向量值 |

一个请求预留 3 个词元、块大小为 2 时，需要两个物理块，容量为 4 个词元。`token_count` 仍可能是 0，因为预留容量不等于已经写入并提交的数据。

```text
physical_blocks = [3, 0]
capacity_tokens = 2 blocks * 2 slots = 4
token_count     = 0 before commit
```

`locate_reserved(2)` 可以返回第二个物理块的第 0 个槽位；`locate(2)` 必须等第 3 个词元提交后才能读取。这个区分阻止 Attention 访问尚未完成的 K/V。

## 分配失败不能留下半成品

`reserve(request_id, total_tokens)` 先计算目标块数，再检查空闲块总量。容量不足时返回 `false`，原页表和空闲计数保持不变。容量足够时才把物理块附加到请求页表。

| 场景 | 返回值 | 页表变化 | 空闲块变化 |
| --- | --- | --- | --- |
| 已有容量足够 | `true` | 不变 | 不变 |
| 需要扩展且容量足够 | `true` | 追加物理块 | 减少 |
| 容量不足 | `false` | 不变 | 不变 |

实现位于 <a href="/_code/chapter07_kv_cache/include/chapter07_kv_cache/block_table.h"><code>block_table.h</code></a> 和 <a href="/_code/chapter07_kv_cache/src/block_table.cpp"><code>block_table.cpp</code></a>。

## 验证容量与失败原子性

```bash
cmake -S code/chapter07_kv_cache -B /tmp/cpp-llm-chapter07-pool \
  -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/cpp-llm-chapter07-pool \
  --target chapter07_block_pool_test -j2
ctest --test-dir /tmp/cpp-llm-chapter07-pool \
  -R '^chapter07_block_pool_test$' --output-on-failure
```

测试覆盖多请求隔离、容量耗尽、提交范围拒绝和失败后空闲块计数不变。下一篇把真实 K/V 数值放入这些物理槽位。
