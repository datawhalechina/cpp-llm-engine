# 09.1 先固定正确结果和工作负载

性能数字只有在输出正确、输入条件相同的前提下才有意义。本章把正确性检查放在计时结果之前：任一 golden、上下文、缓存事务或输出长度检查失败，本次记录只能用于排错，不能进入 A/B 比较。

## 正确性闸门检查什么

| 检查项 | 典型证据 | 失败后的处理 |
| --- | --- | --- |
| 输出 | 固定词元、logits 容差或参考输出 | 标记 `correctness_pass=false` |
| 状态 | position、token history 和 K/V 长度一致 | 停止统计性能结论 |
| 资源 | reserve/commit/rollback 后无泄漏 | 先修复生命周期错误 |
| 批量语义 | 顺序执行与批量执行逐请求一致 | 不比较批量吞吐 |

## 工作负载固定什么

`workload_id` 代表一组可重复的输入条件，不是随意填写的实验名称。改变模型、量化、prompt 集合、输入/输出长度、请求数、批处理策略或后端时，应使用新的 workload 身份或明确的新变体。实验名称回答“这次运行是什么”，工作负载身份回答“它和谁可以比较”。<a href="../../_code/chapter09_performance/include/chapter09_performance/evidence_record.h"><code>EvidenceRecord</code></a> 保存这两类身份以及正确性检查；同一 `workload_id` 下仍需核对环境字段。

| 条件 | 为什么会改变结果 |
| --- | --- |
| 模型与量化 | 权重大小、算术强度和 Kernel 路径不同 |
| Prompt 与输出长度 | Prefill/Decode 占比和 K/V 容量不同 |
| 请求数与到达模式 | 排队、批量机会和尾延迟不同 |
| 后端与线程配置 | 指令集、并行度和内存访问路径不同 |
| 调度与缓存策略 | 每步有效词元数和资源等待不同 |

用于 A/B 的实现开关不应混入 workload 身份。比如 `scalar/AVX2` 和 `sequential/parallel` 描述“怎样执行”，应写入实验身份和检查项；请求数、prompt、输出长度与到达方式描述“执行什么”，才属于 workload。这样同一负载下的两种实现仍然可比。

## 验证正确性闸门

从仓库根目录运行：

```bash
bash code/chapter09_performance/scripts/run.sh
```

`chapter09_evidence_contract_test` 会接受完整记录，并拒绝正确性失败、样本数不等于测量轮数或不完整的变长 prompt。通过后再进入 `09.2`，把同一记录契约放进预热和重复测量流程。
