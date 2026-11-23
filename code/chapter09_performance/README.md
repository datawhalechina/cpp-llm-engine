# Chapter 09 代码：性能证据与优化决策

本目录把一次性能实验整理成可复查证据：固定环境和工作负载，通过正确性检查，保存逐轮样本，再比较实现差异。原理与学习顺序见 [`Chapter09 导读`](../../source/chapter09_performance/next/intro.md)，已归档实验的工作负载、结果路径和适用边界见[实验资产登记表](../../source/reference/experiment_assets.md)。

## 代码架构

Chapter08、Chapter10 或专项 Kernel 入口产生原始样本；Benchmark 记录端到端结果，Profiler 附件解释热点原因。可进入统一证据链的样本由迁移器规范化为 `EvidenceRecord`，专项 A/B 和矩阵实验则保留配对顺序与工作负载维度。`EvidenceRecord::valid()` 检查 schema、环境、工作负载和正确性；`summarize()` 从逐轮样本生成统计值；两份记录有效且可比时，`compare()` 才给出决策。

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
| Q2_K 单 Warp payload 复用能否减少 L1TEX 等待 | `scripts/run_deepseek2_cuda_q2_warp_ab.sh` |
| 固定 IQ4_NL 优化后，Q2_K 收益能否进入完整 Prefill | `scripts/run_deepseek2_cuda_q2_warp_runtime_ab.sh` |
| Q2_K/IQ4_NL 优化后，完整 Prefill 的热点迁移到哪里 | `scripts/run_deepseek2_cuda_optimized_profile.sh` |
| Q3_K 位平面与分组 scale 复用能否改善融合 GEMV | `scripts/run_deepseek2_cuda_q3_warp_ab.sh` |
| 固定 Q2_K/IQ4_NL 优化后，Q3_K 收益能否进入完整 Prefill | `scripts/run_deepseek2_cuda_q3_warp_runtime_ab.sh` |
| 设备 Router 能否替代每层完整 hidden 回传 | `scripts/run_deepseek2_cuda_device_router_ab.sh` |
| 设备 Router 接入后还剩哪些 Host/Device 往返 | `CHAPTER09_DEEPSEEK2_OPTIMIZED_PROFILE_GEMV_PATH=all-warp CHAPTER09_DEEPSEEK2_OPTIMIZED_PROFILE_MOE_PATH=device-router bash scripts/run_deepseek2_cuda_optimized_profile.sh` |
| 设备端专家分组能否替代 Host 建表与小型 H2D | `scripts/run_deepseek2_cuda_device_grouping_ab.sh` |

CUDA 入口需要相应模型、CUDA Toolkit，性能剖析入口还需要 Nsight Systems 或 Nsight Compute。Profiler 附件用于解释原因，不能替代端到端 A/B 决策。已归档 CUDA 结果与默认路径结论见[实验资产登记表](../../source/reference/experiment_assets.md#多模态cuda-与正确性资产)。

DeepSeek Tensor Prefill 剖析入口通过 `CHAPTER09_DEEPSEEK2_TENSOR_PREFILL_PROFILE_MLA_PATH=legacy|absorbed` 显式选择路径，默认值为 `absorbed`。结果元数据会记录该选择，避免把优化前后的 Kernel 分布混在同一结论中。

DeepSeek 量化 GEMV 使用两级证据。格式矩阵先固定权重形状与 batch，分别确认 Q2_K 的 L1TEX 等待和 IQ4_NL 的 barrier 等待；格式专用 A/B 再只改变线程组织，并以 CPU reference、执行指令和吞吐解释变化。真实模型实验逐项叠加优化：EXP-GPU-24 只切换 IQ4_NL，EXP-GPU-26 固定已验证的 IQ4_NL 后只切换 Q2_K。只有完整 27 层 Prefill 的反向顺序配对也改善且 logits 保持一致时，Factory 才切换默认路径。

默认路径改变后，用 `run_deepseek2_cuda_optimized_profile.sh` 重新捕获完整 Prefill。默认 `block-warp + packed` 复现 EXP-GPU-27；设置 `GEMV_PATH=all-warp` 生成 EXP-GPU-30；再设置 `MOE_PATH=device-router` 生成 EXP-GPU-32。三项实验都固定 absorbed MLA，只按登记组合改变 GEMV 或 MoE 路径，用于判断默认值切换后的热点迁移，不与早期 EXP-GPU-20 的 absorbed MLA profile 混用。

EXP-GPU-28/29 对 Q3_K 重复同一门禁：微基准与 NCU 先解释位平面复用，再在完整 Prefill 中固定 Q2_K、IQ4_NL、MLA 和 MoE 路径。Q3_K 端到端收益成立后，Factory 才从 `AdaptiveBlockWarp` 切换到 `AdaptiveAllWarp`。

EXP-GPU-31 固定 absorbed MLA、all-Warp GEMV 和 packed 专家执行，只把 Router/Top-K 从 Host 移到设备。候选路径仍向 Host 回传 `[T,K]` 紧凑结果并建立专家分组，因此该实验只决定路由计算位置；后续 EXP-GPU-32/33 分别完成传输复查和设备分组 A/B。

EXP-GPU-32 表明完整 hidden D2H 已被紧凑路由结果替代：长度 128 的 D2H 字节从 `27.673 MB` 降到 `0.569 MB`。`3117` 次小型 H2D 保持不变，对应 Host 建组后的行号和权重，因此 EXP-GPU-33 继续验证设备建组，而没有融合只占约 `1.0%` Kernel 时间的 Router。

EXP-GPU-33 固定设备 Router，只比较 Host 建组与设备建组。设备候选在长度 32/128 的中位数比例为 `1.225/0.998`，正确性完全一致但没有稳定端到端收益。结果支持保留 `DeviceGroupedDispatch` 作为可复现实验路径，不支持切换 Factory 默认值。

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
