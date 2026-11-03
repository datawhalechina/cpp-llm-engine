# 08.1 建立性能分析基础

Benchmark 回答“快了多少”，profiling 回答“时间花在哪里以及下一步查什么”。学习者先固定工作负载并确认输出正确，再从阶段汇总进入事件、函数或硬件证据，不能根据单个吞吐数字直接猜瓶颈。

## 区分结果测量与原因定位

一次记录至少包含实验 ID、workload ID、模型、输入输出长度、batch、后端、硬件、构建类型和运行轮次。实验 ID 指向证据资产，workload ID 固定可比较条件；缺少后者时，两组数字不能直接比较。

| 问题 | Benchmark | Profiling |
| --- | --- | --- |
| 主要输出 | 延迟、吞吐、内存和正确性 | 阶段耗时、事件、热点和等待点 |
| 回答的问题 | 改动是否产生稳定变化 | 变化可能来自计算、内存、同步还是调度 |
| 工程决定 | 是否值得继续调查 | 下一次单变量实验改哪里 |

## 从阶段进入热点

性能证据按下面的顺序逐步收窄。前一级没有成立时，不应跳到后一级下结论。教学结果中 `prefill=50%`、`decode=25%` 只用于练习阶段拆分，不代表真实模型热点；真实 TinyLlama 的服务和 Decoder 证据在 [`08.8`](./08_real_service_profile.md) 单独分析。

运行默认入口可以从 Chapter07 的 `EXP-CPU-01` 结果选择四请求 round-robin workload，生成阶段报告并完成一次不声明优化收益的同构 A/B 检查。适配器会拒绝缺少或错误的实验身份字段：

```bash
bash code/chapter08_profiling/scripts/run.sh
```

| 观察层级 | 当前教程中的证据 | 能回答什么 |
| --- | --- | --- |
| 固定 workload | JSON 中的 `experiment_id`、`workload_id` 和环境字段 | 证据来自哪里，现象能否复现 |
| 阶段汇总 | input、prefill、decode、sampling、output | 时间主要落在哪一段 |
| 事件与函数 | trace events、hotspot、Decoder 分段计时 | 下一步检查哪个调用或投影 |
| 系统证据 | `perf`、Nsight、硬件计数器 | CPU、内存或 GPU 假设是否成立 |
| 单变量 A/B | baseline、experiment、correctness gate | 改动是否具有因果收益 |

## 验证与衔接

读完后应能说明：当前数字来自哪个 workload，最慢的是哪个阶段，现有证据能否支持具体函数或硬件归因。下一节 [`08.2 延迟拆解`](./02_latency_breakdown.md) 继续定义各阶段的计时起止点；完整证据强度见 [`08.9 Profiling 证据层级`](./09_profiling_evidence_hierarchy.md)。
