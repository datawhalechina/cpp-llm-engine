# FQ15 KV Block Pool

## 任务

实现一个固定容量的 KV Cache block pool。请求申请若干 block，pool 记录 request 到 block 数量的映射；请求结束后释放 block，后续请求可以复用释放出来的空间。

## 完成标准

- 初始化时拒绝零容量 pool。
- request 只能成功申请足够数量的空闲 block。
- 同一个 request 不能重复申请。
- 容量不足时保留已有分配，不产生部分分配。
- 释放未知 request 返回明确结果。
- 释放后 block 可以被新 request 复用。

## 验证

运行 `bash exercises/foundation/run_all.sh`，重点检查 `fq15_kv_block_pool` 的容量不足、释放和复用路径。

## 提示

- L0：先维护每个 block 的空闲状态。
- L1：先检查空闲数量，再修改状态，避免申请失败时留下部分分配。
- L2：用 request ID 保存分配结果，释放时只回收该 request 拥有的 block。

## 主线映射

对应 [`Chapter07 K/V 形状`](/chapter07_kv_cache/next/01_kv_shape)、[`块表与内存池`](/chapter07_kv_cache/next/04_block_table_pool) 和 [`内存管理参考`](/reference/memory_management)。
