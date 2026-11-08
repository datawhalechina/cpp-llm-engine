# Chapter 07 代码：分页 KV Cache

本目录把逻辑词元位置映射到固定大小物理块，并保存每层真实 K/V。先阅读 [`Chapter07 导读`](../../source/chapter07_kv_cache/next/intro.md)，再沿事务接口进入源码。

## 代码架构

`KvCacheTransaction` 是 Chapter08 和 Chapter10 可见的资源接口，提供预留、提交、裁剪、释放和只读定位。`KvBlockPool` 管理 `request_id -> BlockTable`、物理块引用计数与空闲状态；`CpuPagedKvCache` 同时拥有块池、key/value payload 和逐层写入标记。`RadixPrefixCache` 通过 `PrefixBlockStore` 操作 CPU 或 CUDA 块所有权，按词元路径索引完整 K/V 块，让新请求挂接只读前缀后继续写入私有后缀。

```text
Chapter08 / Chapter10
          │ KvCacheTransaction
          ▼
 reserve -> write every layer -> commit -> Attention read
    │              │                │
    └──── failure: trim ─────────────┘
                   │
         terminal: decrement refs -> clear unreferenced blocks
```

预留只增加容量，不让新位置可读；所有层写入成功后才能提交。独占块在最后一个引用释放后才清理 payload；共享前缀块由 Radix 索引和请求页表分别持有引用。Chapter06 产生词元，Chapter08 选择下一步请求，本章不决定调度顺序。

`RadixPrefixCache` 只负责匹配、块引用、命中统计和叶端 LRU 淘汰，不直接修改 Request 或模型 Context。Engine 的 CPU/CUDA Factory 持有索引，并通过 Chapter06 的生命周期钩子完成三方同步；Chapter10 准入时只估算命中，提升运行时才采用真实前缀，终态释放前发布完整块。请求引用计数会固定活跃共享块，容量重试只回收索引独占的叶端块。

## 阅读顺序与资产

| 顺序 | 文件或对象 | 阅读重点 | 对应验证 |
| --- | --- | --- | --- |
| 1 | `block_table.h` | `KvCacheTransaction`、逻辑位置和事务接口 | 两个专项测试共用该契约 |
| 2 | `block_table.cpp` | 块池预留、提交、失败原子性和回收 | `chapter07_block_pool_test` |
| 3 | `paged_kv_cache.cpp` 的写入与提交 | payload 可见性、逐层完成条件和批量原子性 | `chapter07_paged_kv_cache_test` |
| 4 | `trim_to_tokens()`、`release()` | 清理发生在物理块重新可用之前 | 裁剪与释放复用用例 |
| 5 | `attention()` | 根据块表直接读取跨块 K/V，并与连续参考比较 | 跨块 GQA 数值用例 |
| 6 | `radix_prefix_cache.cpp` | 最长块对齐前缀、共享引用和私有后缀 | `chapter07_radix_prefix_cache_test` |
| 7 | `run_real_paged_kv.sh` | TinyLlama 连续 Context 与分页 Context 的交接 | `real_tinyllama_paged_kv_test` |

## 运行与验证

从仓库根目录运行默认 CPU 入口：

```bash
bash code/chapter07_kv_cache/scripts/run.sh
```

CTest 应通过块池、分页数据面和 Radix 前缀缓存三项测试。它们分别固定控制面 contract，验证 payload、Attention 与失败原子性，并证明共享块不会因单个请求结束而提前释放。

已准备 TinyLlama 模型和 llama.cpp CPU 构建时，再运行真实模型验收：

```bash
bash code/chapter07_kv_cache/scripts/run_real_paged_kv.sh
```

输出中的 `paged_kv_real_match: true` 证明固定 prompt 下连续/分页 logits 与 token 一致，并检查错误位置、非法 token、回滚和终态释放；`radix_context_match: true` 证明目标请求采用共享前缀后只计算私有后缀，`radix_eviction_match: true` 证明零空闲块时可回收索引独占后缀并保持结果一致。

## 扩展入口

CUDA 路径已在 `code/cuda` 实现共享分页 payload、直接读取 BlockTable 的 GQA backend 和分页双请求 Decode batch。CPU/CUDA Factory 共用 Radix 匹配与淘汰算法，分别负责主机和设备 payload 的最后引用清理；ServiceCore 准入只估算，提升运行时才同步 `BlockTable`、Request Prefill 进度和真实模型 Context。Chapter08 已能按实际命中长度排序；跨线程锁和分层淘汰策略仍未实现。

当前连续与分页 payload 都保存浮点 K/V。K/V Cache 量化仍需增加每层或每块 scale metadata、低精度 payload、事务一致性检查、Attention 读取 backend 和长上下文误差测试；BlockTable 与引用计数接口应保持不变。
