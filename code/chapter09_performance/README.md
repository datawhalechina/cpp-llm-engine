# Chapter 09 代码：性能证据与优化决策

本目录把一次性能实验整理成可复查证据：固定环境和负载，通过正确性检查，保存逐轮样本，再比较实现差异。原理与学习顺序见 [`Chapter09 导读`](../../source/chapter09_performance/next/intro.md)，已归档实验的负载、结果路径和适用边界见[实验资产登记表](../../source/reference/experiment_assets.md)。

## 代码架构

Chapter08、Chapter10 或专项计算核入口产生原始样本；性能基准记录端到端结果，性能剖析附件解释热点原因。导入器把已有样本转换为 `EvidenceRecord`，专项 A/B 和矩阵实验保留配对顺序与负载维度。`EvidenceRecord::valid()` 检查格式、环境、负载和正确性；`summarize()` 从逐轮样本生成统计值；两份记录有效且可比时，`compare()` 给出决策。

```text
Runtime / Scheduler / Service / Kernel
              │
      ┌───────┴────────┐
      ▼                ▼
Benchmark 样本      Profiler 附件
端到端与保护指标    阶段 / Kernel / 硬件原因
      │                │
      └───────┬────────┘
              ▼
正确性 + 身份 + 配对顺序 + 适用范围矩阵
              │
              ▼
EvidenceRecord / 专项决策 -> KEEP / REJECT / 重测
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
| 2 | `evidence_record.h/.cpp` | 环境、负载和正确性字段 | `chapter09_evidence_contract_test` |
| 3 | `comparison.h/.cpp` | 可比性、主指标和保护指标 | `chapter09_comparison_test` |
| 4 | `import_legacy_evidence.mjs` | 旧结果怎样进入统一格式 | `chapter09_legacy_import_test` |
| 5 | `results/raw/`、`results/evidence/` | 原始样本和统一记录 | `chapter09_versioned_evidence_test` |
| 6 | `results/benchmark/`、`results/profile/` | A/B 结果与性能剖析附件 | 实验资产登记表 |

比较前必须确认两份记录各自有效，并且模型、后端、设备、硬件、构建信息与负载一致。主指标改善用于判断收益，延迟、吞吐、内存和正确性保护指标用于阻止以明显回退换取单项提升。

## 运行默认验证

从仓库根目录运行：

```bash
bash code/chapter09_performance/scripts/run.sh
```

脚本运行统计、记录格式、旧结果导入、真实样本转换和比较决策测试。它验证数据处理流程；真实性能由下面的专项入口测量。

旧结果保留逐轮样本时，可以转换为统一记录：

```bash
node code/chapter09_performance/scripts/import_legacy_evidence.mjs \
  --input code/chapter09_performance/results/legacy/chapter07_cpu_baseline.json \
  --output-dir /tmp/chapter09-evidence
```

迁移器要求输入保留逐轮样本；只有均值或 P50 的历史文件继续作为只读参考。

## 选择实验入口

先根据问题选择入口，并区分局部计算核、完整运行时、调度和服务四层证据。

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

这些入口采用固定负载、正确性闸门和反向执行顺序。具体实验编号、样本数量、结果数值与结论范围只在[实验资产登记表](../../source/reference/experiment_assets.md#cpu-runtimedecoder-与服务实验)维护。

### CUDA

| 要回答的问题 | 入口 |
| --- | --- |
| 端到端时间主要落在哪些 Kernel | `scripts/run_cuda_profile.sh` |
| 三阶段与融合在线 Attention 怎样比较 | `scripts/run_cuda_attention_ab.sh` |
| 分页 Tensor Prefill 是否优于逐词元路径 | `scripts/run_cuda_paged_prefill_ab.sh` |
| 双请求张量组批是否优于顺序解码 | `scripts/run_cuda_batch_ab.sh` |
| 分页融合 Attention 是否值得替换默认路径 | `scripts/run_cuda_paged_attention_ab.sh` |
| Q4_K/Q6_K GEMV 怎样随 batch 扩展 | `scripts/run_cuda_quantized_gemv_matrix.sh` |
| 自适应 tile 的局部原因指标是否成立 | `scripts/run_cuda_adaptive_gemv_ab.sh` |
| 自适应 tile 是否改善真实 Decode | `scripts/run_cuda_adaptive_gemv_runtime_ab.sh` |
| 收益能否跨 batch、上下文和 Decode 步数保持 | `scripts/run_cuda_adaptive_gemv_runtime_matrix.sh` |
| DeepSeek 跨请求 grouped MoE 是否改善完整 CUDA Decode | `scripts/run_deepseek2_cuda_grouped_moe_ab.sh` |
| DeepSeek grouped Decode 的时间落在哪些 Kernel 与同步点 | `scripts/run_deepseek2_cuda_grouped_moe_profile.sh` |
| DeepSeek 分页 MLA 相对连续 MLA 的端到端代价是多少 | `scripts/run_deepseek2_cuda_paged_mla_ab.sh` |
| DeepSeek 整段 Tensor Prefill 相对逐词元调用能减少多少时间 | `scripts/run_deepseek2_cuda_tensor_prefill_ab.sh` |
| DeepSeek 长 Prefill 在指定 MLA 路径下的时间落在哪里 | `CHAPTER09_DEEPSEEK2_TENSOR_PREFILL_PROFILE_MLA_PATH=absorbed bash scripts/run_deepseek2_cuda_tensor_prefill_profile.sh` |
| MLA 权重吸收能否降低长度 32/128 的完整 Prefill 时间 | `scripts/run_deepseek2_cuda_absorbed_mla_ab.sh` |
| 设备端行收集与加权散回能否替代 MoE 逐行复制 | `scripts/run_deepseek2_cuda_moe_dispatch_ab.sh` |
| Q2_K 与 IQ4_NL 的 batch 扩展和首要停顿原因是什么 | `scripts/run_deepseek2_cuda_gemv_matrix.sh` |
| 32 线程 IQ4_NL 是否优于通用 256 线程归约 | `scripts/run_deepseek2_cuda_iq4_warp_ab.sh` |
| IQ4_NL 单 Warp 收益能否进入完整 27 层 Prefill | `scripts/run_deepseek2_cuda_iq4_warp_runtime_ab.sh` |
| Q2_K 单线程束载荷复用能否减少 L1TEX 等待 | `scripts/run_deepseek2_cuda_q2_warp_ab.sh` |
| 固定 IQ4_NL 优化后，Q2_K 收益能否进入完整 Prefill | `scripts/run_deepseek2_cuda_q2_warp_runtime_ab.sh` |
| Q2_K/IQ4_NL 优化后，完整 Prefill 的热点迁移到哪里 | `scripts/run_deepseek2_cuda_optimized_profile.sh` |
| Q3_K 位平面与分组 scale 复用能否改善融合 GEMV | `scripts/run_deepseek2_cuda_q3_warp_ab.sh` |
| 固定 Q2_K/IQ4_NL 优化后，Q3_K 收益能否进入完整 Prefill | `scripts/run_deepseek2_cuda_q3_warp_runtime_ab.sh` |
| 设备 Router 能否替代每层完整 hidden 回传 | `scripts/run_deepseek2_cuda_device_router_ab.sh` |
| 设备 Router 接入后还剩哪些 Host/Device 往返 | `CHAPTER09_DEEPSEEK2_OPTIMIZED_PROFILE_GEMV_PATH=all-warp CHAPTER09_DEEPSEEK2_OPTIMIZED_PROFILE_MOE_PATH=device-router bash scripts/run_deepseek2_cuda_optimized_profile.sh` |
| 设备端专家分组能否替代 Host 建表与小型 H2D | `scripts/run_deepseek2_cuda_device_grouping_ab.sh` |

CUDA 入口需要相应模型、CUDA Toolkit，性能剖析入口还需要 Nsight Systems 或 Nsight Compute。Profiler 附件用于解释原因，不能替代端到端 A/B 决策。已归档 CUDA 结果与默认路径结论见[实验资产登记表](../../source/reference/experiment_assets.md#多模态cuda-与正确性资产)。

首次使用 `nsys` 或 `ncu` 时，先阅读 [`09.5 按问题选择性能剖析工具`](../../source/chapter09_performance/next/05_profiling_tools.md)。该节给出工具检查、报告导出、指标阅读和从计算核原因返回端到端 A/B 的完整顺序；本页只维护稳定的代码入口。

DeepSeek Tensor Prefill 剖析入口通过 `CHAPTER09_DEEPSEEK2_TENSOR_PREFILL_PROFILE_MLA_PATH=legacy|absorbed` 显式选择路径，默认值为 `absorbed`。结果元数据会记录该选择，避免把优化前后的 Kernel 分布混在同一结论中。

DeepSeek 量化 GEMV 使用两级证据。格式矩阵固定权重形状与批大小，定位 Q2_K 和 IQ4_NL 的主要停顿；格式专用 A/B 只改变线程组织，并以 CPU 参考结果、执行指令和吞吐解释变化。真实模型实验再逐项叠加已经验证的优化，完整 27 层预填充保持 logits 一致且端到端时间改善后，工厂才切换默认路径。设备 Router、设备端专家分组和热点迁移的实验编号、配置与结果见[实验资产登记表](../../source/reference/experiment_assets.md)。

### 多模态

| 要回答的问题 | 入口 | 当前证据 |
| --- | --- | --- |
| CPU 动态切图、视觉编码、语言生成和服务的端到端时间怎样分布 | `code/multimodal/scripts/run_real_service_benchmark.sh` | 重复样本、阶段统计、固定输出和资源释放 |
| CUDA 视觉流水线与图文生成是否保持 CPU 语义 | `code/multimodal/scripts/run_cuda_smolvlm_context.sh ... mixed-prompt` | 完整图文序列、四词元生成和 CPU/CUDA 对照 |
| CUDA 服务能否接收图像请求并组批 Prefill/Decode | `code/multimodal/scripts/run_cuda_smolvlm_context.sh ... mixed-service` | JSON/Base64 图片、同步/SSE、双请求组批和终态释放 |
| Packed Prefill 是否优于串行 Context Prefill | `code/chapter09_performance/scripts/run_cuda_smolvlm_prefill_ab.sh` | 两个等长图文请求共同执行 Linear/FFN，Attention 保持请求级 K/V 隔离；两轮端到端均回退，因此只保留显式实验路径 |

现有 CUDA 多模态证据回答正确性、组合关系、请求隔离和语言 Prefill 路径选择。视觉阶段与语言阶段仍需建立分阶段端到端性能 A/B；新增结论时应固定图片、切图数量、文本提示和输出长度，再分别记录视觉预处理、Vision Encoder、Projector、预填充与解码样本。

## 结果目录

| 路径 | 保存内容 | 是否参与决策 |
| --- | --- | --- |
| `results/raw/` | 未聚合的逐轮样本 | 是 |
| `results/evidence/` | 统一 schema 与比较决策 | 是 |
| `results/benchmark/` | 专项 A/B 和矩阵实验 | 通过正确性与可比性后参与 |
| `results/profile/` | nsys、ncu 和阶段归因附件 | 只解释原因 |
| `results/legacy/` | 重构前历史结果 | 仅追溯 |

默认脚本把临时结果写入 `/tmp`，仓库中的基线保持不变。归档结果的字段和登记位置见[实验资产登记表](../../source/reference/experiment_assets.md)。

## 扩展入口

新增性能入口时按以下顺序建设：

1. 定义负载、baseline、候选实现、主指标和保护指标。
2. 让两条路径通过相同的正确性闸门。
3. 保存逐轮样本，并用反向顺序或独立进程控制执行顺序影响。
4. 从局部计算逐层复核到 Decoder、Runtime、Scheduler 或 Service。
5. 用矩阵说明结论覆盖的 batch、上下文和 Decode 步数。
6. 将结果登记到实验资产表，并在本页保留稳定入口。

CPU、CUDA、多模态和服务实验复用同一证据结构；平台专属 profiler 只负责归因，不改变统一比较规则。
