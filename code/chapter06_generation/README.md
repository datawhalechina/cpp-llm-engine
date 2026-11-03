# Chapter 06：Generation Runtime 生成运行时

本目录把 Decoder 结构连接成 `prefill -> decode -> sampling -> output` 的最小 Generation Runtime，并显式管理 Request、KV Cache、Scheduler、token-budget step 和 streaming。服务 API、限流和资源池由 Chapter09 承接。完整学习说明见 [`source/chapter06_generation`](../../source/chapter06_generation/intro.md)。

## 本章资产总览

| 部分 | 职责 |
| --- | --- |
| 生成核心 | `prefill -> decode -> sampling -> output`、Request、KV Cache、基础 Scheduler 和 Streaming |
| 批量与服务预览 | 教学 ragged Tensor batch、token-level Continuous Batching、chunked prefill、公平性、物理 block ID 和 BlockTable；队列与服务接入由 Chapter09 承接 |

| 资产 | 作用 |
| --- | --- |
| `src/request.cpp` | Request 状态、取消和超时 |
| `src/kv_cache.cpp` | Request 可观察的 token history；不保存 Attention K/V 数值 |
| `src/embedding.cpp`、`position_encoding.cpp` | 输入 embedding 和位置编码 |
| `src/transformer.cpp`、`decoder_layer.cpp` | prefill/decode 的模型结构入口 |
| `RaggedBatch`、`TensorBatchScheduler` | 扁平值与行偏移组成的教学 CPU batch；一次调用推进多个请求 |
| `src/sampler.cpp` | logits 到 token 的采样 |
| `src/scheduler.cpp` | 多请求 round-robin 推进 |
| `include/chapter06_generation/model_executor.h` | 每请求完整/分块 prefill、decode 与只读模型状态快照 |
| `src/serving_preview.cpp` | token/block budget、`KvBlockPool`、BlockTable 与三类状态一致性检查 |
| `src/paged_kv_cache.cpp` | CPU 分层 physical block payload、逐层提交、分页 GQA 读取与释放清空 |
| `ExecutorBatchInput` | 将真实 executor 的 token、position、request 与预留 KV 写入槽对应起来 |
| `src/runtime.cpp` | 状态、cache、sampler 的运行时编排 |
| `tests/` | generation、scheduler、Tensor batch、连续/分页 KV Cache、sampler、Serving Preview 和 smoke 测试 |
| `examples/main_generation.cpp` | 普通生成 demo |
| `examples/main_streaming.cpp` | streaming callback demo |
| `scripts/run.sh` | 隔离构建、测试和运行两个 demo |

## 运行与验证

从仓库根目录执行：

```bash
bash code/chapter06_generation/scripts/run.sh
```

验证重点是请求状态、完整/分块 prefill、Tensor batch 与逐请求 golden、公平推进、分页 K/V 写读、block 释放复用和 streaming 输出。`CpuPagedKvCache` 已验证跨 block GQA 与连续参考输出一致，但真实 TinyLlama Context 仍使用连续 K/V。`ExecutorBatchInput` 与 ServiceCore 也尚未接入真实 batch 或分页 payload；GPU batch 和 GPU PagedAttention 仍未完成。
