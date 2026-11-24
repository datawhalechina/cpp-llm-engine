# 10.8 从请求时间线产生指标

指标来自明确的状态转换。<a href="https://github.com/datawhalechina/cpp-llm-engine/blob/main/code/chapter10_serving/src/service_core.cpp"><code>ServiceCore</code></a> 使用单调时钟记录准入、进入运行状态、首词元和终态；`HttpAdapter` 把这些事实序列化为 Prometheus 文本。

接受、终态、在途和释放计数的守恒关系见 [`F-SERVE-01`](../../reference/formulas.md#f-serve-01)。

## 核心计数与延迟

| 指标 | 更新位置 | 回答的问题 |
| --- | --- | --- |
| `accepted/queue_full/duplicate` | 准入 | 请求是否进入服务 |
| `waiting/running/peak_inflight` | 准入、提升与释放 | 实例压力有多大 |
| `generated_tokens` | 调度器返回的模型步统计 | 服务完成多少生成工作 |
| cancelled/timed_out/failed | 统一释放时按 Request 终态计数 | 请求为何停止 |
| released | 统一释放完成 | 资源是否真正归还 |
| 排队时间 | 准入 → 运行 | 容量或调度等待多久 |
| TTFT | 准入 → 首词元 | 用户多久看到响应 |
| 完成时间 | 准入 → 终态 | 请求多久结束 |

## 指标、日志和链路追踪各自回答什么

| 证据 | 适合回答 | 最小关联键 |
| --- | --- | --- |
| 指标 | 总量、速率和延迟分布 | service/model/负载标签 |
| 结构化日志 | 单次错误和状态转换 | request ID |
| 链路追踪（Trace） | 跨传输层、调度器和执行器的耗时 | trace/span ID |

教学实现先提供计数器和 `_sum/_count`。P50/P99 来自直方图或 Chapter09 的原始样本；总和与次数只能计算均值。

## 验证指标口径

运行 `bash code/chapter10_serving/scripts/run.sh`。`chapter10_service_core_test` 应核对 accepted、running、generated_tokens、terminal 和 released 计数；`chapter10_http_contract_test` 应确认 metrics 端点只序列化这些事实。需要中位数或尾延迟时，将原始样本交给 Chapter09，不在服务端从总和反推。`10.9` 使用真实模型核对整条时间线。
