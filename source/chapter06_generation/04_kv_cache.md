# 06.4 区分词元历史、模型 K/V 与 BlockTable

生成过程需要保存历史状态，但仓库里有五个名称相近、用途不同的对象。阅读时从 Attention 计算进入 K/V 形状，再沿请求生命周期进入物理 block ownership。

## 五种缓存对象

Chapter06 的 `KVCache` 只保存 Request 已消费和生成的 token ID。它让教学 Runtime 可以检查请求历史和生命周期，但不保存 Attention 的 key/value tensor。

| 对象 | 保存内容 | 所有者 | 代码入口 |
| --- | --- | --- | --- |
| Chapter06 `KVCache` | `TokenSequence`，即 token ID 历史 | `Request` | <a href="../_code/chapter06_generation/include/chapter06_generation/kv_cache.h"><code>kv_cache.h</code></a> |
| Chapter04 `KvCache` | 单层连续 `float` key/value，布局为 `[tokens, heads, head_dim]` | Attention 教学测试 | <a href="../_code/chapter04_attention/include/chapter04_attention/kv_cache.h"><code>kv_cache.h</code></a> |
| `RealLlamaCpuContext` | 每层真实 K/V 数值、position 和执行统计 | 单个真实模型请求 | <a href="../_code/model/include/cpp_llm_engine/model/real_llama_cpu_decoder.h"><code>real_llama_cpu_decoder.h</code></a> |
| `KvBlockPool` / `BlockTable` | 物理 block ID、逻辑 token 到 block/slot 的映射 | Scheduler / ServiceCore | <a href="../_code/chapter06_generation/include/chapter06_generation/serving_preview.h"><code>serving_preview.h</code></a> |
| `CpuPagedKvCache` | 分层物理 block K/V payload 和分页 GQA 读取 | Chapter06 教学数据面 | <a href="../_code/chapter06_generation/include/chapter06_generation/paged_kv_cache.h"><code>paged_kv_cache.h</code></a> |

## 从 Attention 看序列长度增长

设 prefill 后已有 `S` 个 token，每层 K/V 的逻辑形状是 `[S, Hkv, Dh]`。一次 decode 只计算当前 token 的 `k_new` 和 `v_new`，形状为 `[1, Hkv, Dh]`；提交成功后，缓存的逻辑形状变为 `[S+1, Hkv, Dh]`，当前 query 对 `S+1` 个位置计算分数。

“拼接历史 K/V”描述的是逻辑结果，不应默认一定发生一次大数组复制。TinyLlama CPU 连续实现用 `vector::insert` 把新行追加到每层存储；完整 Paged KV 则把新行写入 `BlockTable` 指定的 block/slot，Attention 按页表读取历史，不需要先把离散块拼成连续数组。

| 阶段 | 新输入 | 提交后的 K/V 形状 | Attention 读取范围 |
| --- | --- | --- | --- |
| Prefill | `[S, Hkv, Dh]` | `[S, Hkv, Dh]` | 每个 query 只读取因果范围内的位置 |
| 第一次 decode | `[1, Hkv, Dh]` | `[S+1, Hkv, Dh]` | 位置 `0..S` |
| 第 `g` 次 decode | `[1, Hkv, Dh]` | `[S+g, Hkv, Dh]` | 位置 `0..S+g-1` |

教学 `KVCache` 的公开操作只有追加 token、清空、读取长度和读取 token 序列。下面代码对应仓库中的真实接口，不代表分层 K/V tensor。

```cpp
KVCache cache;
cache.append(token_id);
const TokenSequence& history = cache.tokens();
cache.clear();
```

`Request` 持有这个对象。Runtime 成功推进后更新 token history；请求结束、取消、超时或失败后，ServiceCore 的统一释放路径会清空它。

## 从连续 K/V 进入分块管理

`RealLlamaCpuModel` 只保存共享只读权重。每个请求通过 `create_context()` 获得独立的 `RealLlamaCpuContext`，Context 为每层保存 key/value 数值并维护当前位置。

| 操作 | Context 行为 | logits 行为 |
| --- | --- | --- |
| 首个 prefill chunk | `reset()` 后逐 token 写入各层 K/V | 只有 final chunk 请求最后一个 token 的 logits |
| 后续 prefill chunk | 在已有 position 和分层 K/V 后追加 | 中间 chunk 不产生 logits |
| decode | 读取历史 K/V，追加一个 token 的 K/V | 产生下一次采样所需 logits |
| reset / 请求释放 | position 归零并清空每层 key/value | 不保留上一个请求状态 |

`KvBlockPool` 根据 token 数预留物理 block ID，`BlockTable` 把逻辑 token 位置映射成 `{physical_block, slot}`。固定大小 block 让请求按需增加容量并在终态直接归还 block，避免每次序列增长都重新申请一整段更大的连续缓存。`CpuPagedKvCache` 已使用这些位置保存教学 K/V；真实 `RealLlamaCpuContext` 仍持有连续 K/V，因此真实模型尚未获得分页布局的容量收益。

| 操作 | 作用 | 失败时要求 |
| --- | --- | --- |
| `reserve(request_id, total_tokens)` | 为目标 token 数预留足够 block | 容量不足时不做部分修改 |
| `commit_tokens(...)` | 提交已经成功执行的 token 数 | 不能超过已预留容量 |
| `trim_to_tokens(...)` | executor 未推进时撤销多余预留 | 保留已提交 token 对应 block |
| `locate(...)` | 返回逻辑 token 的物理 block 和 slot | 越界必须明确失败 |
| `release(request_id)` | 请求终态时归还全部 block | 不影响其他请求的 block table |

## 状态变化与验证

Runtime 同时更新 Request 可见状态和真实 Context，但两者承担不同职责。Scheduler 只决定本 step 推进哪些 token，不直接读写 K/V 数值。

| 阶段 | Request token history | 真实 Model Context | BlockTable |
| --- | --- | --- | --- |
| Prefill 开始 | 保存已提交 prompt token | 建立各层 K/V 和 position | 预留并提交 prompt token 容量 |
| Chunked prefill | `prefill_offset` 逐段前进 | 在原 Context 后追加 K/V | 每段成功后提交 token count |
| Decode | 追加采样得到的 token | 用历史 K/V 计算并追加新 K/V | 为新 token 提交位置 |
| 终态释放 | `clear()` | Executor/Context 析构或 reset | `release()` 归还物理 block |

三个计数不会在每个时刻都相等。prefill 或 decode 先把输入 token 写入模型 K/V，再从 logits 采样一个输出 token；新输出已经进入 Request history，但要到下一次 decode 才进入模型 K/V 和 BlockTable。

| 时刻 | Token history | 模型 K/V 已提交 token | BlockTable token |
| --- | --- | --- | --- |
| 中间 prefill，已提交 `P` 个 prompt token | `P` | `P` | `P` |
| final prefill 后采样首 token | `S + 1` | `S` | `S` |
| 已生成 `G` 个 token | `S + G` | `S + G - 1` | `S + G - 1` |

`RequestExecutionState` 使用 `prefill_offset + max(generated_size - 1, 0)` 计算模型应已消费的 token 数，再分别核对 `RequestModelExecutor::execution_state()` 和 BlockTable。真实 Executor 只返回 token count、KV 层数和是否保存 K/V，不向 Scheduler 暴露缓存数值。

按“Attention 读写 → `Seq_Len` 增长 → 逻辑位置映射 → block 回收”的顺序阅读，可以把算子、数据和系统三个层次连接到同一次 decode。

| 验证对象 | 测试入口 | 通过标准 |
| --- | --- | --- |
| Chapter06 token history | <a href="../_code/chapter06_generation/tests/kv_cache_test.cpp"><code>kv_cache_test.cpp</code></a> | append、size、tokens 和 clear 正确 |
| Chapter04 连续 K/V | <a href="../_code/chapter04_attention/tests/chapter04_attention_test.cpp"><code>chapter04_attention_test.cpp</code></a> | append、容量拒绝、decode offset 和 clear 正确 |
| `KvBlockPool` | <a href="../_code/chapter06_generation/tests/block_pool_scheduling_test.cpp"><code>block_pool_scheduling_test.cpp</code></a> | reserve、commit、locate、release 和容量受限调度正确 |
| `CpuPagedKvCache` | <a href="../_code/chapter06_generation/tests/paged_kv_cache_test.cpp"><code>paged_kv_cache_test.cpp</code></a> | 跨 block 输出与连续参考一致，逐层提交和释放复用正确 |
| 三类状态推进 | <a href="../_code/chapter06_generation/tests/execution_transaction_test.cpp"><code>execution_transaction_test.cpp</code></a> | prefill、decode 和失败回滚的计数关系正确 |
| 真实分层 K/V | `real_tinyllama_generation_test`、`real_tinyllama_serving_test` | golden token、chunked prefill、Context 隔离和终态释放同时通过 |
