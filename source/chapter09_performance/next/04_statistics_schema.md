# 09.4 保存样本，再生成统计结果

单次运行容易受到缓存、频率和后台负载影响。结果文件应保留原始样本，聚合值用于快速阅读，二者不能互相替代。

## 最小统计集合

| 字段 | 用途 |
| --- | --- |
| `count` | 确认样本量 |
| `min` / `max` | 观察波动范围和异常值 |
| `mean` | 观察总体平均成本 |
| `median` | 降低少量异常值对典型结果的影响 |
| `p90` / `p99` | 样本量足够时观察尾部，不用于很小的教学样本 |

## 统一证据记录

测量记录只保存事实；A/B 决策记录引用两份测量，再保存策略和结论。这样修改回退阈值时，不需要重写原始实验结果。

| 测量记录部分 | 必要内容 |
| --- | --- |
| 身份 | `schema_version`、`experiment_id`、`workload_id` |
| 环境 | model、backend、device、build、compiler |
| 工作负载 | 请求数、输入/输出长度、执行模式、预热与迭代数 |
| 正确性 | 是否通过及检查项 |
| 样本 | 每轮总时间、阶段时间、吞吐和内存 |
| 聚合 | count、min、max、mean、median |

同一批请求的 prompt 长度可以不同。长度一致时使用 `prompt_tokens_per_request`；长度不一致时使用与请求一一对应的 `prompt_tokens` 数组，例如 `[6, 1]`。比较器会把该数组纳入可比性检查，避免把不同输入规模的实验放进同一结论。

| 对比记录部分 | 必要内容 |
| --- | --- |
| 引用 | baseline 与 experiment 的身份 |
| 可比性 | workload 和关键环境字段是否一致 |
| 策略 | 显式主指标，以及延迟、吞吐和内存允许的回退阈值 |
| 变化 | 延迟、吞吐和阶段差异 |
| 决策 | `KEEP`、`REJECT` 或 `NOT_COMPARABLE` |

## 验证样本与聚合值

<a href="../../_code/chapter09_performance/src/statistics.cpp"><code>summarize()</code></a> 对样本副本排序，计算 count、min、max、mean 和 median；<a href="../../_code/chapter09_performance/src/evidence_record.cpp"><code>EvidenceRecord::summary()</code></a> 把同一规则应用到每个指标。执行：

```bash
bash code/chapter09_performance/scripts/run.sh
```

`chapter09_statistics_test` 应通过空样本、奇数样本、偶数样本和输入顺序无关性检查；`chapter09_evidence_contract_test` 应确认 JSON 同时保留身份、原始样本和聚合值。`09.5` 在这份记录之外增加更细的热点证据。
