# 10.8 从请求时间线产生指标

指标应来自明确的状态转换，而不是在 HTTP 层估算。<a href="../../_code/chapter10_serving/src/service_core.cpp"><code>ServiceCore</code></a> 使用单调时钟记录 admission、进入 running、首词元和终态，Transport 额外记录连接与写回错误。

## 核心计数与延迟

| 指标 | 更新位置 | 回答的问题 |
| --- | --- | --- |
| accepted/queue_full/duplicate | admission | 请求是否进入服务 |
| waiting/running/peak_inflight | 状态迁移 | 实例压力有多大 |
| generated_tokens | token callback | 服务完成多少生成工作 |
| cancelled/timed_out/failed | 终态写入 | 请求为何停止 |
| released | 统一释放完成 | 资源是否真正归还 |
| queue wait | admission → running | 容量或调度等待多久 |
| TTFT | admission → first token | 用户多久看到响应 |
| completion | admission → terminal | 请求多久结束 |

## 指标、日志和 Trace 各自回答什么

| 证据 | 适合回答 | 最小关联键 |
| --- | --- | --- |
| Metrics | 总量、速率和延迟分布 | service/model/workload 标签 |
| 结构化日志 | 单次错误和状态转换 | request ID |
| Trace | 跨 Transport、Scheduler 和 Executor 的耗时 | trace/span ID |

教学实现先提供 counter 和 `_sum/_count`；P50/P99 需要 histogram 或 Chapter09 的原始样本，不能从只有总和与次数的数据中恢复。

## 验证指标口径

运行 `bash code/chapter10_serving/scripts/run.sh`。`chapter10_service_core_test` 应核对 accepted、running、generated_tokens、terminal 和 released 计数；`chapter10_http_contract_test` 应确认 metrics 端点只序列化这些事实。需要中位数或尾延迟时，将原始样本交给 Chapter09，不在服务端从总和反推。`10.9` 使用真实模型核对整条时间线。
