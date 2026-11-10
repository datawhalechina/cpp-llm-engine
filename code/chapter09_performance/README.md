# Chapter 09 代码：性能证据与优化决策

本目录把一次性能实验整理成可复查证据：固定环境和工作负载，通过正确性检查，保存逐轮样本，再比较实现差异。原理与学习顺序见 [`Chapter09 导读`](../../source/chapter09_performance/next/intro.md)，已归档实验的工作负载、结果路径和适用边界见[实验资产登记表](../../source/reference/experiment_assets.md)。

## 代码架构

Chapter08 或 Chapter10 产生原始样本，迁移器将结果规范化为 `EvidenceRecord`。`EvidenceRecord::valid()` 检查 schema、环境、工作负载和正确性；`summarize()` 从逐轮样本生成统计值；两份记录有效且可比时，`compare()` 才给出决策。

```text
原始样本
   │
   ▼
EvidenceRecord::valid ──> summarize ──> compare
   │                         │             │
   └─ 身份与正确性           └─ 聚合值      └─ KEEP / REJECT /
                                             NOT_COMPARABLE / INVALID
```

| 阶段 | 输入 | 代码入口 | 输出 |
| --- | --- | --- | --- |
| 规范化 | 服务或批执行原始 JSON | `import_legacy_evidence.mjs` | evidence v2 JSON |
| 校验 | 单份 evidence | `EvidenceRecord::valid()` | 有效记录或字段错误 |
| 聚合 | 逐轮 `MeasurementSample` | `summarize()` | 数量、范围、均值和中位数 |
| 比较 | baseline、experiment 与策略 | `compare()` | 独立决策记录 |

## 阅读顺序与资产

| 顺序 | 资产 | 阅读重点 | 验证 |
| --- | --- | --- | --- |
| 1 | `statistics.h/.cpp` | 统计值怎样从原始样本产生 | `chapter09_statistics_test` |
| 2 | `evidence_record.h/.cpp` | 环境、工作负载和正确性字段 | `chapter09_evidence_contract_test` |
| 3 | `comparison.h/.cpp` | 可比性、主指标和保护指标 | `chapter09_comparison_test` |
| 4 | `import_legacy_evidence.mjs` | 旧结果怎样进入统一格式 | `chapter09_legacy_import_test` |
| 5 | `results/raw/`、`results/evidence/` | 原始样本和统一记录 | `chapter09_versioned_evidence_test` |
| 6 | `results/benchmark/`、`results/profile/` | A/B 结果与性能剖析附件 | 实验资产登记表 |

比较前必须确认两份记录各自有效，并且模型、后端、设备、硬件、构建信息与工作负载一致。主指标改善用于判断收益，延迟、吞吐、内存和正确性保护指标用于阻止以明显回退换取单项提升。

## 运行默认验证

从仓库根目录运行：

```bash
bash code/chapter09_performance/scripts/run.sh
```

脚本运行统计、证据格式、版本迁移、真实证据桥接和比较决策测试。它验证证据链代码，不测量当前机器的真实性能。

旧结果保留逐轮样本时，可以转换为统一记录：

```bash
node code/chapter09_performance/scripts/import_legacy_evidence.mjs \
  --input code/chapter09_performance/results/legacy/chapter07_cpu_baseline.json \
  --output-dir /tmp/chapter09-evidence
```

只有均值或 P50 的历史文件不能还原原始样本，迁移器会拒绝这类输入。

## 选择实验入口

先根据问题选择入口，不要把局部 Kernel、完整 Runtime、调度和服务结果混为同一层证据。

### CPU 与 MoE

| 要回答的问题 | 入口 |
| --- | --- |
| 真实服务样本能否进入统一 evidence | `scripts/run_real_evidence.sh service` |
| 全层权重缓存怎样交换时间与常驻内存 | `scripts/run_qwen15_moe_layer_cache.sh` |
| 按专家分组能否减少重复权重解码 | `scripts/run_qwen15_moe_grouped_batch_ab.sh` |
| 分组执行能否穿过完整 Decoder | `scripts/run_qwen15_moe_batch_runtime_ab.sh` |
| DeepSeek MLA + MoE 完整 Decode 是否受益于专家分组 | `scripts/run_deepseek2_moe_batch_runtime_ab.sh` |
| 收益怎样随 batch 和 Decode 步数变化 | `scripts/run_qwen15_moe_batch_runtime_matrix.sh` |
| 收益能否穿过固定调度队列 | `scripts/run_qwen15_moe_scheduling_ab.sh` |
| 收益能否穿过动态请求接纳和终态释放 | `scripts/run_qwen15_moe_service_ab.sh` |

这些入口采用固定工作负载、正确性闸门和反向执行顺序。具体实验编号、样本数量、结果数值与结论范围只在[实验资产登记表](../../source/reference/experiment_assets.md#cpu-runtimedecoder-与服务实验)维护。

### CUDA

| 要回答的问题 | 入口 |
| --- | --- |
| 端到端时间主要落在哪些 Kernel | `scripts/run_cuda_profile.sh` |
| 三阶段与融合在线 Attention 怎样比较 | `scripts/run_cuda_attention_ab.sh` |
| 分页 Tensor Prefill 是否优于逐词元路径 | `scripts/run_cuda_paged_prefill_ab.sh` |
| 双请求 Tensor Batch 是否优于顺序 Decode | `scripts/run_cuda_batch_ab.sh` |
| 分页融合 Attention 是否值得替换默认路径 | `scripts/run_cuda_paged_attention_ab.sh` |
| Q4_K/Q6_K GEMV 怎样随 batch 扩展 | `scripts/run_cuda_quantized_gemv_matrix.sh` |
| 自适应 tile 的局部原因指标是否成立 | `scripts/run_cuda_adaptive_gemv_ab.sh` |
| 自适应 tile 是否改善真实 Decode | `scripts/run_cuda_adaptive_gemv_runtime_ab.sh` |
| 收益能否跨 batch、上下文和 Decode 步数保持 | `scripts/run_cuda_adaptive_gemv_runtime_matrix.sh` |

CUDA 入口需要相应模型、CUDA Toolkit，性能剖析入口还需要 Nsight Systems 或 Nsight Compute。Profiler 附件用于解释原因，不能替代端到端 A/B 决策。已归档 CUDA 结果与默认路径结论见[实验资产登记表](../../source/reference/experiment_assets.md#多模态cuda-与正确性资产)。

## 结果目录

| 路径 | 保存内容 | 是否参与决策 |
| --- | --- | --- |
| `results/raw/` | 未聚合的逐轮样本 | 是 |
| `results/evidence/` | 统一 schema 与比较决策 | 是 |
| `results/benchmark/` | 专项 A/B 和矩阵实验 | 通过正确性与可比性后参与 |
| `results/profile/` | nsys、ncu 和阶段归因附件 | 只解释原因 |
| `results/legacy/` | 重构前历史结果 | 仅追溯 |

默认脚本把临时结果写入 `/tmp`，不会覆盖仓库基线。需要归档新结果时，先登记新的实验与工作负载身份，再提交原始样本、环境信息、正确性结果和决策文件。

## 扩展入口

新增性能入口时按以下顺序建设：

1. 定义工作负载、baseline、候选实现、主指标和保护指标。
2. 让两条路径通过相同的正确性闸门。
3. 保存逐轮样本，并用反向顺序或独立进程控制执行顺序影响。
4. 从局部计算逐层复核到 Decoder、Runtime、Scheduler 或 Service。
5. 用矩阵说明结论覆盖的 batch、上下文和 Decode 步数。
6. 将结果登记到实验资产表，README 只增加稳定入口。

CPU、CUDA、多模态和服务实验复用同一证据结构；平台专属 profiler 只负责归因，不改变统一比较规则。
