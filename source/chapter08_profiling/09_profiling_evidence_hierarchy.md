# 08.9 Profiling 证据层级

性能数字只有在正确性、workload 和测量边界明确后才能用于决策。证据按“结果可信 → 现象可复现 → 阶段可定位 → 热点可解释 → 改动有因果”逐级收敛；高层证据不能补救低层缺失。

## 五级证据链

| 层级 | 必须回答的问题 | 本仓库资产 | 不足时不能声称 |
| --- | --- | --- | --- |
| E0 正确性闸门 | 输出、状态和资源是否正确 | golden token、Context 隔离、terminal release | 任何性能收益有效 |
| E1 可重复 workload | 模型、输入、backend、硬件和轮次是否固定 | Chapter07/09 JSON 与环境字段 | 两次运行可直接比较 |
| E2 墙钟现象 | TTFT、completion、throughput 是否稳定变化 | sequential/parallel、scalar/optimized 多轮统计 | 为什么变快或变慢 |
| E3 阶段归因 | 时间落在 prefill、decode、Linear、Attention 还是服务编排 | `ProfileRecord`、Decoder 分段计时 | 具体函数是热点 |
| E4 系统热点 | CPU 符号、cache miss、GPU kernel 或同步点是否支持假设 | `perf`、Nsight Systems/Compute、硬件计数器 | 硬件瓶颈已经确认 |
| E5 单变量 A/B | 只改变一个实现后，正确性和目标指标是否同时通过 | Q4_K/Q6_K backend A/B、回归门 | 改动具有因果收益 |

E4 依赖环境权限；无法采集 `perf` 或 Nsight 时，应明确停在 E3，而不是用阶段计时冒充符号或硬件证据。E5 可以基于 E3 形成实验，但热点解释的可信度仍受 E4 缺失限制。

## 三类并发证据

| 执行方式 | 可观察证据 | 当前状态 |
| --- | --- | --- |
| StepBatch 选择 | 一个 step 选择多个 request/token | Chapter06 已实现，模型仍逐 candidate 调用 |
| Request parallel | 多个独立 Executor/Context 在 CPU worker 上重叠 | Chapter09 已实现，真实 A/B 已归档 |
| Tensor batch | 一次 Decoder/算子调用消费合并后的 batch Tensor | 教学 Tensor batch 与独立真实 scalar Linear batch 已实现；ServiceCore 尚未接入，固定实验为 `0.8541x` |

请求级并发中，两个 Executor 的累计 compute time 可能大于墙钟时间。`overlapped_compute = max(aggregate_compute - dispatch, 0)` 只证明计时区间发生重叠，不等于 CPU 利用率、并行效率或 Tensor batch 加速比。真实 scalar batch 的独立配对结果见 [`07.8 服务负载评测`](../chapter07_benchmark/08_service_workloads.md#scalar-batch-配对结果)。

## 从证据到决定

一次可保留的优化至少包含 E0、E1、E2、E3 和 E5：先通过 golden 与资源检查，再固定 workload，确认墙钟变化，定位阶段，最后做单变量 A/B。具备系统 profiler 时补 E4，用符号、kernel 或硬件计数器解释为什么该阶段受益。

原始结果和复现入口统一登记在[实验与验证资产索引](../reference/experiment_assets.md)。正文只引用实验 ID、结论和适用范围，不复制未经环境标注的临时数字。
