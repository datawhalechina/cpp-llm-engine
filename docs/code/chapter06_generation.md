# chapter06_generation 代码导航

本章代码包含 Request、Scheduler、KV Cache、Sampler、Runtime 和端到端测试。

## 两部分入口

- [生成循环](../chapter06_generation/03_naive_generation.md)
- [连续批处理](../chapter06_generation/10_continuous_batching.md)
- [真实模型批量执行](../chapter06_generation/11_real_model_batch.md)
- [分块 KV Cache](../chapter06_generation/12_block_kv_cache.md)

## 阅读顺序

1. 先看 `request.cpp` 和 `runtime.cpp`
2. 再看 `kv_cache.cpp`、`scheduler.cpp`、`sampler.cpp`
3. 最后看测试和脚本

## 运行时骨架

| 文件 | 作用 |
| --- | --- |
| <a href="/_code/chapter06_generation/include/chapter06_generation/request.h"><code>request.h</code></a> | 请求状态接口 |
| <a href="/_code/chapter06_generation/src/request.cpp"><code>src/request.cpp</code></a> | 请求生命周期 |
| <a href="/_code/chapter06_generation/include/chapter06_generation/runtime.h"><code>runtime.h</code></a> | Runtime 接口 |
| <a href="/_code/chapter06_generation/src/runtime.cpp"><code>src/runtime.cpp</code></a> | 生成 runtime |
| <a href="/_code/chapter06_generation/src/kv_cache.cpp"><code>src/kv_cache.cpp</code></a> | KV Cache |
| <a href="/_code/chapter06_generation/src/scheduler.cpp"><code>src/scheduler.cpp</code></a> | Scheduler |
| <a href="/_code/chapter06_generation/src/sampler.cpp"><code>src/sampler.cpp</code></a> | Sampler |
| <a href="/_code/chapter06_generation/src/transformer.cpp"><code>src/transformer.cpp</code></a> | Transformer 组装 |

## 验证与脚本

| 文件 | 作用 |
| --- | --- |
| <a href="/_code/chapter06_generation/examples/main_generation.cpp"><code>examples/main_generation.cpp</code></a> | 生成示例 |
| <a href="/_code/chapter06_generation/examples/main_streaming.cpp"><code>examples/main_streaming.cpp</code></a> | 流式示例 |
| <a href="/_code/chapter06_generation/tests/request_contract_test.cpp"><code>tests/request_contract_test.cpp</code></a> | Request 状态机契约 |
| <a href="/_code/chapter06_generation/tests/chapter06_generation_test.cpp"><code>tests/chapter06_generation_test.cpp</code></a> | Generation Core 端到端测试 |
| <a href="/_code/chapter06_generation/tests/execution_transaction_test.cpp"><code>tests/execution_transaction_test.cpp</code></a> | 预留、执行和回滚事务 |
| <a href="/_code/chapter06_generation/tests/kv_block_pool_contract_test.cpp"><code>tests/kv_block_pool_contract_test.cpp</code></a> | BlockPool 容量守恒、块独占和失败原子性 |
| <a href="/_code/chapter06_generation/scripts/run.sh"><code>scripts/run.sh</code></a> | 运行脚本 |

## 对应文档

- [chapter06 导读](../chapter06_generation/intro.md)
