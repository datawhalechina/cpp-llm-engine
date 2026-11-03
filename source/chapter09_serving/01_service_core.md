# 09.1 构建 C++ 服务核心

`ServiceCore` 把 Chapter06 的 `Request`、Runtime、Sampler 和可选 Executor 放进准入、等待、运行和释放流程。它不改变模型数学，而是决定请求何时进入执行、一次推进多少 token，以及终态后释放哪些资源。

## 运行并阅读最小闭环

从仓库根目录执行默认入口：

```bash
bash code/chapter09_serving/scripts/run.sh
```

示例最多容纳 3 个请求，每轮最多向 Runtime 提交 2 个请求。先观察输出中的 admission、completed、released 和 inflight，再按接口、实现、测试的顺序阅读源码。

| 阅读顺序 | 源码 | 先回答什么 |
| --- | --- | --- |
| 1 | <a href="../_code/chapter09_serving/include/chapter09_serving/serving_core.h"><code>serving_core.h</code></a> | ServiceCore 保存哪些请求和指标状态 |
| 2 | <a href="../_code/chapter09_serving/src/serving_core.cpp"><code>serving_core.cpp</code></a> | admission、dispatch 和 release 怎样推进 |
| 3 | <a href="../_code/chapter09_serving/tests/chapter09_serving_test.cpp"><code>chapter09_serving_test.cpp</code></a> | 容量、终态和资源释放如何验证 |

## 请求怎样进入模型执行

`dispatch_step(runtime, max_batch_tokens)` 按 token budget 选择 prefill chunk 或 decode；配置 `KvBlockPool` 后，模型调用前还要预留物理 block。真实 Factory 只加载一份 `RealLlamaCpuModel`，每个请求创建独立 `RealLlamaCpuContext`，终态释放 Executor、Context、请求 cache 和 block ID。

请求级并发把独立 Executor/Context 分配给 CPU worker，不合并激活，也不与 block-aware step 混用。Chapter06 已验证的真实 batch 尚未接入 ServiceCore。

| 执行入口 | 怎样推进请求 | 当前适用范围 |
| --- | --- | --- |
| `dispatch()` | 单线程 Round-robin 跑到终态 | 最小同步闭环 |
| `dispatch_step()` | 每 step 按 token/block budget 动态选择 | chunked prefill、BlockTable 控制面 |
| `dispatch_parallel_requests()` | 多个独立请求在 CPU worker 重叠 | 请求级并发，不是 Tensor batch |

## 指标怎样对应时间线

Queue wait 从 admission 到进入 running，TTFT 到第一次 token callback，completion 到 finished、failed、cancelled 或 timed out。延迟使用单调时钟并导出 `_sum/_count`，测试通过手动时钟固定结果。

| 指标 | 观察时机 |
| --- | --- |
| `accepted/queue_full/duplicate_total` | admission 结果 |
| `cancelled/timed_out/failed_total` | 请求进入对应终态 |
| `completed/released_total` | 完成与资源回收 |
| `generated_tokens_total`、`peak_inflight` | token 产生和并发峰值 |
| `queue_wait/ttft/completion_seconds_sum/count` | 三段在线延迟 |

![Chapter 09 在线延迟时间线](../assets/diagrams/chapter09-latency-timeline.svg)

## 验证与衔接

默认测试把 ServiceCore 的状态推进拆成五个入口。先运行本章脚本，再根据失败标签进入对应测试；真实模型综合入口 `bash scripts/run_cpu_service_learning.sh` 继续检查双请求 golden、Context 隔离、HTTP socket 和 benchmark。HTTP 路由与状态码留到下一节 [`09.2 Transport 与网络库`](./02_transport.md)。

| 服务行为 | 测试入口 | 验证事实 |
| --- | --- | --- |
| 准入与同步生命周期 | <a href="../_code/chapter09_serving/tests/chapter09_serving_test.cpp"><code>chapter09_serving_test.cpp</code></a> | 容量、重复 ID、失败、超时和终态释放 |
| token/block budget | <a href="../_code/chapter09_serving/tests/step_dispatch_test.cpp"><code>step_dispatch_test.cpp</code></a> | 动态补位、BlockPool 容量和完成后复用 |
| 请求级 CPU 并行 | <a href="../_code/chapter09_serving/tests/parallel_dispatch_test.cpp"><code>parallel_dispatch_test.cpp</code></a> | 两个 Executor 重叠且 token 不串请求 |
| 部分 Prefill 取消 | <a href="../_code/chapter09_serving/tests/cancellation_release_test.cpp"><code>cancellation_release_test.cpp</code></a> | request cache 与物理 block 一起释放 |
| 在线延迟指标 | <a href="../_code/chapter09_serving/tests/service_metrics_test.cpp"><code>service_metrics_test.cpp</code></a> | 手动时钟固定 queue wait、TTFT 和 completion 口径 |
