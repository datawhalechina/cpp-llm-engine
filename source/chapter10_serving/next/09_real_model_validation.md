# 10.9 用真实模型验证服务闭环

确定性 Executor 适合覆盖状态机，真实 TinyLlama 用于验证 Model 共享、Context 隔离、Tokenizer、K/V 增长和 HTTP 输出。<a href="../../_code/engine/scripts/run_real_http_server.sh"><code>run_real_http_server.sh</code></a> 组装这些对象；真实模型检查仍先通过 correctness，再记录性能。

## 真实对象怎样组装

| 生命周期 | 对象 | 验证事实 |
| --- | --- | --- |
| 进程启动 | Tokenizer、共享 Model、Executor Factory | 权重只加载一次 |
| 请求创建 | 独立 Executor/Context | token history 和 K/V 不串请求 |
| 服务执行 | ServiceCore → Chapter08 → batch runner | golden token、批量调用与阶段一致 |
| HTTP 写回 | JSON 或 SSE | token 顺序和 finish reason 正确 |
| 请求终态 | release | Context、K/V 和服务条目归零 |

## 分层运行真实验证

| 入口 | 覆盖范围 | 不代表什么 |
| --- | --- | --- |
| 真实 ServiceCore smoke | Model/Context、双请求 batch 和生命周期 | 网络端到端性能 |
| HTTP `--check` | 串行 Adapter 入口下的 socket、Completion、SSE 和 metrics | 并发 HTTP 服务或高并发压测 |
| 服务 Benchmark | 固定 workload 的 TTFT、吞吐和内存 | 开放环生产负载 |
| sequential/parallel A/B | 请求级 CPU 重叠 | Tensor batch 收益 |

归档结果使用 Chapter09 的 workload ID、环境和正确性字段。正文解释验证方法，具体机器上的数字进入实验资产索引。

双请求 ServiceCore smoke 在 admission 时绑定请求 ID 和分页数据面，再把 Decode 候选交给 `RealLlamaExecutorFactory::decode_batch()`。验证同时检查 batch 调用计数、逐请求 golden、`paged_payload_active`、Context 隔离、BlockPool 提交数和终态释放。请求级并行 Worker 仍使用独立连续 Context；它用于比较 CPU 请求重叠，不等同于共享模型的 Tensor batch。

## 验证真实服务闭环

确认 TinyLlama GGUF 已存在后运行：

```bash
bash code/engine/scripts/run_real_http_server.sh --check
```

通过条件是 health、同步 Completion、SSE、metrics 和固定 token golden 全部成功，两个请求的 Context/K/V 不串扰，终态后服务条目和分页块归零。`10.10` 将这个单进程闭环映射到成熟服务架构。
