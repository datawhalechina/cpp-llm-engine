# 09.3 测量真实 ServiceCore

真实服务 benchmark 固定 TinyLlama、双请求和输出长度，测量 `RealLlamaExecutorFactory -> ServiceCore -> Scheduler -> Runtime`。它不经过 HTTP transport，因此回答模型执行和服务编排各占多少时间，不代表网络端到端延迟。

## 固定工作负载与指标

从仓库根目录运行：

```bash
bash code/engine/scripts/run_real_serving_benchmark.sh
```

脚本使用 Release、warmup 1 和测量 3 轮，同时提交 `Hello TinyLlama` 与 `Hi`，共享一份 22 层 Model，各自持有 Context 并生成 4 个 token。性能数据进入比较前必须通过 golden、cache 隔离和 Context 释放。

离线 TTFT 从 dispatch 起算，在线 TTFT 从 admission 起算并包含 queue wait，两种口径不能直接相减。在线 `/metrics` 保存累计 `_sum/_count`，离线 JSON 保存逐轮样本、均值、P50 和范围。

| 指标 | 计时范围 | 回答的问题 |
| --- | --- | --- |
| model load | 进程启动到模型可用 | 冷启动成本 |
| prefill/decode compute | Executor 调用累计 | 模型计算成本 |
| runtime overhead | dispatch 减去模型计算 | Scheduler、Sampler 和状态更新成本 |
| TTFT/completion | dispatch 到首个/最后一个 token | 请求可见延迟 |
| throughput | 生成 token 数除以 dispatch 墙钟 | 固定场景稳态产出 |
| model weight payload | 已加载 Tensor payload 之和 | 共享 Model 的权重载荷规模 |
| committed KV | 请求释放前的分层 K/V 实际字节数 | 两个 Context 已提交多少缓存 |
| process peak RSS | 进程生命周期常驻集高水位 | 本次运行的整体内存上界 |

JSON 的 `memory_bytes` 同时保存两个请求的 committed token、各自 KV 字节数和总量。benchmark 用模型的层数、KV 头数和头维度反算预期 K/V；字节数不一致时 correctness 失败。峰值 RSS 还包含 tokenizer、临时工作区、分配器和运行库，不能与权重、KV 当作互斥分类相加。

## 比较三种执行策略

Scalar 基线、请求级并发和 optimized AVX2 回答不同问题，不能混成同一项“batch 加速”。请求级并发让两个独立 Executor 在 CPU worker 上重叠；optimized policy 切换 Q4_K/Q6_K Linear backend；两者都没有让 ServiceCore 调用 Chapter06 的真实 batch。

详细环境、原始 JSON、逐轮样本和复现脚本集中在[实验与验证资产索引](../reference/experiment_assets.md)。当前归档数据只适合同一 CPU、模型、输入和构建条件下的 A/B。

| 证据 | 当前结果 | 可以得出的结论 |
| --- | --- | --- |
| `EXP-CPU-02` Scalar 服务基线 | dispatch `7559.42 ms`，模型计算占 `99.9979%` | 瓶颈主要在 CPU Decoder，不在 ServiceCore 队列 |
| `EXP-CPU-09` Request parallel | dispatch `-30.38%`，吞吐 `+43.64%`，correctness PASS | 两个独立 Executor 的计时间隔发生重叠 |
| `EXP-CPU-08` Optimized AVX2 | ServiceCore dispatch `-78.4838%`，吞吐 `+364.7483%` | 已验证的量化 Linear backend 组合在当前机器有效 |
| `EXP-CPU-10` Scalar Linear batch | 配对加速比 `0.8541x`，correctness PASS | 独立批量数据路径正确，但尚未接入 ServiceCore，也没有性能收益 |

## 验证与衔接

复现实验时先检查 workload 和环境字段，再检查 correctness，最后比较阶段与墙钟指标。`prefill_compute + decode_compute` 在请求并发模式下可能大于 dispatch 墙钟，只表示计时间隔重叠，不能解释为 Runtime 开销或 Tensor batch 收益。性能原因继续由 [`Chapter08`](../chapter08_profiling/08_real_service_profile.md) 的阶段与单变量证据解释。
