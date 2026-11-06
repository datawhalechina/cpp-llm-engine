# 09.0 用证据判断性能变化

## 本章导读

前八章已经提供模型执行、缓存和调度路径。本章先固定输入、输出和运行环境，再测量延迟、吞吐与内存；发现变化后，用 Profiling 缩小范围，最后通过单变量 A/B 实验决定保留或回退修改。Chapter09 不进入运行时调用链：Chapter08 和 Chapter10 产生样本，本章定义怎样记录、比较和解释这些样本。

| 章节 | 运行时角色 | 主要产物 | 核心验证 |
| --- | --- | --- | --- |
| Chapter08 | 在一个模型 step 内选择请求并组织批量执行 | `StepBatchPlan`、优先级策略、batch runner 交接 | 候选顺序、缓存事务、批量结果 |
| Chapter09 | 对执行和服务结果建立可复查证据 | workload、样本、统计、A/B 决定 | 记录完整、实验可比、结论可回归 |
| Chapter10 | 管理请求进入、连接、终态和资源释放 | ServiceCore、HTTP/SSE、指标 | admission、取消、写回、释放和真实服务 |

| 学习线 | 本章怎样学习 | 可验证产物 |
| --- | --- | --- |
| A 测量 | 从正确性闸门进入固定工作负载，记录样本并聚合指标 | 可复现的测量记录 |
| B 归因与决策 | 从阶段耗时进入热点分析，再做单变量 A/B | 可比较的实验记录与 KEEP/REJECT 结论 |

| 本章新术语 | 先这样理解 | 首次解释 |
| --- | --- | --- |
| 工作负载（workload） | 一组固定的模型、输入、输出长度和并发条件 | [`09.1 正确性与工作负载`](./01_correctness_workload.md) |
| 首词元延迟（TTFT） | 从请求进入到第一个输出词元可用的时间 | [`09.3 指标`](./03_metrics.md) |
| 词元间延迟（ITL） | 流式生成中相邻输出词元之间的时间 | [`09.3 指标`](./03_metrics.md) |
| 性能剖析（Profiling） | 把总耗时拆到阶段、函数或 Kernel | [`09.5 工具顺序`](./05_profiling_tools.md) |

![Chapter 09 主图：从正确性闸门到性能决策](../../assets/diagrams/chapter09-performance.svg)

## 开始学习

先运行 Chapter09 独立入口：

```bash
bash code/chapter09_performance/scripts/run.sh
```

脚本构建证据记录、统计和比较器，运行六个 CTest。通过时会同时证明 schema 校验、旧结果迁移、版本化 evidence、真实样本桥接、基础统计和四类 A/B 决策没有被破坏。

| 顺序 | 路线 | 页面 | 先看源码 | 读完后应得到什么 |
| --- | --- | --- | --- | --- |
| 1 | A | [`09.1 正确性与工作负载`](./01_correctness_workload.md)、[`09.2 实验入口`](./02_benchmark_setup.md) | <a href="../../_code/chapter09_performance/src/evidence_record.cpp"><code>evidence_record.cpp</code></a> | 哪些条件必须固定，错误结果为何不能进入性能比较 |
| 2 | A | [`09.3 指标`](./03_metrics.md)、[`09.4 统计与记录`](./04_statistics_schema.md) | <a href="../../_code/chapter09_performance/src/statistics.cpp"><code>statistics.cpp</code></a> | 原始样本怎样变成可复查的聚合结果 |
| 3 | B | [`09.5 工具顺序`](./05_profiling_tools.md)、[`09.6 热点归因`](./06_hotspot_attribution.md) | 旧实验结果与阶段计时入口 | 应该先看总量、阶段、函数还是 Kernel |
| 4 | B | [`09.7 单变量 A/B`](./07_ab_experiment.md)、[`09.8 决策与回归`](./08_decision_regression.md) | <a href="../../_code/chapter09_performance/src/comparison.cpp"><code>comparison.cpp</code></a> | 如何判断两次实验可比并形成保留或回退结论 |
| 5 | 扩展 | [`09.9 框架对照`](./09_framework_comparison.md) | 生态卡 | 教学证据链与成熟框架工具链的对应关系 |

## 验证与下一步

| 验收层级 | 执行入口 | 通过条件 | 本章要求 |
| --- | --- | --- | --- |
| L1 数据契约 | `bash code/chapter09_performance/scripts/run.sh` | 记录校验、统计聚合、不可比输入和回退策略测试通过 | 必做 |
| L2 旧证据迁移 | `node code/chapter09_performance/scripts/import_legacy_evidence.mjs ...` | 保留逐轮样本的旧结果转换成功，聚合历史结果被拒绝 | 迁移期间必做 |
| L3 真实模型 | `bash code/chapter09_performance/scripts/run_real_evidence.sh service` | TinyLlama 原始样本转换为统一 evidence，正确性闸门通过 | 有模型时必做 |
| L4 服务 A/B | `bash code/chapter09_performance/scripts/run_real_evidence.sh ab` | 保留顺序与并行原始样本、统一 evidence，并输出 `real-service-ab-decision.json` | 进入 Chapter10 后完成 |
