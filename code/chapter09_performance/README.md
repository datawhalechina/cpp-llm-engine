# Chapter 09 代码资产：性能证据与优化决策

本目录统一原 Benchmark 与 Profiling 使用的实验身份、工作负载、正确性、样本和 A/B 决策。学习说明见 [`Chapter09 导读`](../../source/chapter09_performance/next/intro.md)。先运行：

```bash
bash code/chapter09_performance/scripts/run.sh
```

## 资产总览

| 资产 | 职责 | 验证 |
| --- | --- | --- |
| `statistics.h/.cpp` | 从原始样本计算范围、均值和中位数 | `chapter09_statistics_test` |
| `evidence_record.h/.cpp` | 校验并输出统一测量记录 | `chapter09_evidence_contract_test` |
| `comparison.h/.cpp` | 检查可比性并形成四类决策 | `chapter09_comparison_test` |
| `import_legacy_evidence.mjs` | 迁移保留逐轮样本的旧 Benchmark 与服务结果 | `chapter09_legacy_import_test` |
| `results/raw/`、`results/evidence/` | 四组真实服务样本及其统一证据 | `chapter09_versioned_evidence_test` |
| `scripts/run_real_evidence.sh` | 生成真实服务 evidence、A/B 决策或 Profiling 归因附件 | 输出 `chapter09_real_evidence=PASS` |

旧教学 benchmark 和真实 batch 文件保留了逐轮样本，可以转换为统一记录：

```bash
node code/chapter09_performance/scripts/import_legacy_evidence.mjs \
  --input code/chapter09_performance/results/legacy/chapter07_cpu_baseline.json \
  --output-dir /tmp/chapter09-evidence
```

旧服务结果若只有 mean、P50 等聚合值，导入器会拒绝迁移，因为聚合值不能还原逐轮样本。新的 `real_tinyllama_serving_benchmark` 已写出 `samples`，重新运行后即可迁移。Chapter09 库只转换和比较结果；`run_real_evidence.sh` 会调用 Engine 脚本生成本地真实模型样本。

当前四组 TinyLlama scalar/AVX2、顺序/并行请求原始样本位于 `results/raw/`，转换后的统一证据位于 `results/evidence/`。重构前的实现和结果保存在 `legacy/` 与 `results/legacy/`，只用于追溯，不是默认学习入口。

有本地模型时，运行一次不覆盖仓库基线的真实服务实验：

```bash
bash code/chapter09_performance/scripts/run_real_evidence.sh service
```

需要阶段 Profiling 或顺序/并行请求 A/B 时，将模式改为 `profile`、`ab` 或 `all`。`profile` 产生用于热点归因的文本附件，不参与 A/B 决策；`service` 和 `ab` 才生成统一 evidence。
