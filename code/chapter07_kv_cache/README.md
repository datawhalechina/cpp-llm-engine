# Chapter 07 代码：分页 KV Cache

本目录把逻辑词元位置映射到固定大小物理块，并保存每层真实 K/V。先阅读 [`Chapter07 导读`](../../source/chapter07_kv_cache/next/intro.md)，再沿事务接口进入源码。

## 代码架构

`KvCacheTransaction` 是 Chapter08 和 Chapter10 可见的资源接口，提供预留、提交、裁剪、释放和只读定位。`KvBlockPool` 只管理 `request_id -> BlockTable` 与物理块空闲状态；`CpuPagedKvCache` 同时拥有块池、key/value payload 和逐层写入标记。

```text
Chapter08 / Chapter10
          │ KvCacheTransaction
          ▼
 reserve -> write every layer -> commit -> Attention read
    │              │                │
    └──── failure: trim ─────────────┘
                   │
         terminal: clear payload -> release blocks
```

预留只增加容量，不让新位置可读；所有层写入成功后才能提交。裁剪和释放在请求仍拥有物理块时先清理 payload 与写入标记，再归还块，避免新请求继承旧状态。Chapter06 产生词元，Chapter08 选择下一步请求，本章不决定调度顺序。

## 阅读顺序与资产

| 顺序 | 文件或对象 | 阅读重点 | 对应验证 |
| --- | --- | --- | --- |
| 1 | `block_table.h` | `KvCacheTransaction`、逻辑位置和事务接口 | 两个专项测试共用该契约 |
| 2 | `block_table.cpp` | 块池预留、提交、失败原子性和回收 | `chapter07_block_pool_test` |
| 3 | `paged_kv_cache.cpp` 的写入与提交 | payload 可见性、逐层完成条件和批量原子性 | `chapter07_paged_kv_cache_test` |
| 4 | `trim_to_tokens()`、`release()` | 清理发生在物理块重新可用之前 | 裁剪与释放复用用例 |
| 5 | `attention()` | 根据块表直接读取跨块 K/V，并与连续参考比较 | 跨块 GQA 数值用例 |
| 6 | `run_real_paged_kv.sh` | TinyLlama 连续 Context 与分页 Context 的交接 | `real_tinyllama_paged_kv_test` |

## 运行与验证

从仓库根目录运行默认 CPU 入口：

```bash
bash code/chapter07_kv_cache/scripts/run.sh
```

CTest 应通过 `chapter07_block_pool_test` 和 `chapter07_paged_kv_cache_test`。前者提供控制面 contract 证据；后者提供 payload、Attention、失败原子性和复用隔离的 component 证据。

已准备 TinyLlama 模型和 llama.cpp CPU 构建时，再运行真实模型验收：

```bash
bash code/chapter07_kv_cache/scripts/run_real_paged_kv.sh
```

输出中的 `paged_kv_real_match: true` 证明固定 prompt 下连续/分页 logits 与 token 一致，并检查错误位置、非法 token、回滚和终态释放。

## 扩展入口

GPU 后端可以保留 `KvCacheTransaction` 与 BlockTable 语义，替换 payload 布局和分页 Attention kernel。前缀复用需要在块所有权上增加共享引用、命中与淘汰；多请求选择仍由 Chapter08 负责，多模态输入在进入语言 Decoder 后复用同一缓存事务。
