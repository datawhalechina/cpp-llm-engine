# Chapter 09 代码：性能证据与优化决策

本章把一次性能实验整理成可复查的证据：先固定环境和工作负载，再通过正确性检查，保留逐轮样本，最后才比较优化前后的结果。先阅读 [`Chapter09 导读`](../../source/chapter09_performance/next/intro.md)，再运行：

```bash
bash code/chapter09_performance/scripts/run.sh
```

## 代码架构

Chapter08 或 Chapter10 产生原始样本，迁移器把现有结果规范化为 `EvidenceRecord`。`EvidenceRecord::valid()` 先检查 schema、环境、workload 和正确性，`summarize()` 再从逐轮样本生成统计值；只有两份记录都有效且可比时，`compare()` 才给出性能决策。

| 阶段 | 输入 | 代码对象 | 输出 |
| --- | --- | --- | --- |
| 规范化 | 服务或 batch 原始 JSON | `import_legacy_evidence.mjs` | evidence v2 JSON |
| 校验 | 单份 evidence | `EvidenceRecord::valid()` | 有效记录或字段错误 |
| 聚合 | 逐轮 `MeasurementSample` | `summarize()` | count、范围、均值和中位数 |
| 比较 | baseline、experiment 与策略 | `compare()` | `KEEP`、`REJECT`、`NOT_COMPARABLE` 或 `INVALID` |

## 阅读顺序与资产

| 模块 | 学习内容 | 验证 |
| --- | --- | --- |
| `statistics.h/.cpp` | 从逐轮样本计算范围、均值和中位数 | `chapter09_statistics_test` |
| `evidence_record.h/.cpp` | 校验 schema、硬件身份、工作负载、正确性和原始样本 | `chapter09_evidence_contract_test` |
| `comparison.h/.cpp` | 先检查可比性，再根据主指标和保护指标给出决策 | `chapter09_comparison_test` |
| `import_legacy_evidence.mjs` | 迁移仍保留逐轮样本的旧实验 | `chapter09_legacy_import_test` |
| `results/raw/`、`results/evidence/` | 保存真实原始样本和统一证据 | `chapter09_versioned_evidence_test` |
| `results/profile/`、A/B decision | 保存带 workload 身份的阶段结果和独立决策记录 | `chapter09_versioned_evidence_test` |
| `run_real_evidence.sh` | 生成服务实验、A/B 决策或 Profiling 附件 | `chapter09_real_evidence_bridge_test` |

1. 阅读 `EvidenceRecord`，明确一份证据必须包含哪些环境、负载和正确性字段。
2. 阅读 `summarize()`，理解统计值如何从逐轮样本得到，而不是从另一组聚合值推测。
3. 阅读 `compare()`，区分 `INVALID`、`NOT_COMPARABLE`、`KEEP` 和 `REJECT`。
4. 最后查看 `results/raw/` 与 `results/evidence/`，把数据结构对应到真实实验文件。

比较前必须同时满足两项条件：两份记录各自有效，并且模型、后端、设备、硬件、构建信息与工作负载一致。主指标改善只决定优化是否有收益；延迟、吞吐和内存保护阈值还会阻止以另一项明显退化换取单项提升。

## 运行与验证

默认脚本运行统计、证据格式、版本迁移和比较决策测试：

```bash
bash code/chapter09_performance/scripts/run.sh
```

### 旧结果迁移

旧教学 benchmark 和真实 batch 文件保留了逐轮样本，可以转换为统一记录：

```bash
node code/chapter09_performance/scripts/import_legacy_evidence.mjs \
  --input code/chapter09_performance/results/legacy/chapter07_cpu_baseline.json \
  --output-dir /tmp/chapter09-evidence
```

旧服务结果若只有 mean、P50 等聚合值，导入器会拒绝迁移，因为聚合值不能还原逐轮样本。新的 `real_tinyllama_serving_benchmark` 会写出 `samples`，重新运行后即可迁移。

当前四组 TinyLlama scalar/AVX2、顺序/并行请求原始样本位于 `results/raw/`，转换后的统一证据位于 `results/evidence/`。重构前的实现和结果保存在 `legacy/` 与 `results/legacy/`，只用于追溯，不是默认学习入口。

### 真实实验

有本地模型时，运行一次不覆盖仓库基线的真实服务实验：

```bash
bash code/chapter09_performance/scripts/run_real_evidence.sh service
```

需要阶段 Profiling 或顺序/并行请求 A/B 时，将模式改为 `profile`、`ab` 或 `all`。`profile` 产生用于热点归因的文本附件，不参与 A/B 决策；`service` 和 `ab` 才生成统一证据。

## 扩展入口

新增性能入口时先定义 workload 和正确性检查，再输出逐轮样本并转换为统一 evidence。CPU、CUDA、多模态和服务实验复用同一证据结构；平台专属 profiler 附件只负责归因，不替代 A/B 决策记录。
