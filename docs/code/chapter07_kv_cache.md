# chapter07_kv_cache 代码导航

本章代码包含请求块表、物理 KV 块池和 CPU 分页 K/V 数据面。

## 阅读顺序

1. 先看 `block_table.h/.cpp`，跟踪逻辑词元到物理槽位
2. 再看 `paged_kv_cache.h/.cpp`，跟踪每层 K/V 写入与读取
3. 最后看两个测试，核对事务、跨块结果和释放复用

## 控制面与数据面

| 文件 | 作用 |
| --- | --- |
| <a href="https://github.com/datawhalechina/cpp-llm-engine/blob/main/code/chapter07_kv_cache/include/chapter07_kv_cache/block_table.h"><code>block_table.h</code></a> | `BlockTable`、`KvBlockPool` 和位置接口 |
| <a href="https://github.com/datawhalechina/cpp-llm-engine/blob/main/code/chapter07_kv_cache/src/block_table.cpp"><code>block_table.cpp</code></a> | 预留、提交、裁剪和释放 |
| <a href="https://github.com/datawhalechina/cpp-llm-engine/blob/main/code/chapter07_kv_cache/include/chapter07_kv_cache/paged_kv_cache.h"><code>paged_kv_cache.h</code></a> | CPU 分页 K/V 公共接口 |
| <a href="https://github.com/datawhalechina/cpp-llm-engine/blob/main/code/chapter07_kv_cache/src/paged_kv_cache.cpp"><code>paged_kv_cache.cpp</code></a> | 分层 payload、提交检查和分页 Attention |

## 验证与脚本

| 文件 | 作用 |
| --- | --- |
| <a href="https://github.com/datawhalechina/cpp-llm-engine/blob/main/code/chapter07_kv_cache/tests/block_pool_test.cpp"><code>block_pool_test.cpp</code></a> | 容量、页表定位和失败原子性 |
| <a href="https://github.com/datawhalechina/cpp-llm-engine/blob/main/code/chapter07_kv_cache/tests/paged_kv_cache_test.cpp"><code>paged_kv_cache_test.cpp</code></a> | 分层提交、跨块结果和释放复用 |
| <a href="https://github.com/datawhalechina/cpp-llm-engine/blob/main/code/chapter07_kv_cache/scripts/run.sh"><code>scripts/run.sh</code></a> | Chapter07 独立构建与 CTest 入口 |
