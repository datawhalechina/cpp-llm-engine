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
| `run_cuda_profile.sh` | 用 Nsight Systems 生成真实 CUDA 四词元生成的 kernel/memory 基线 | 固定 golden token、环境元数据与 CSV 附件 |
| `run_cuda_attention_ab.sh` | 比较三阶段与融合在线 Attention 的真实 Tensor Prefill/Decode | 两轮独立运行、逐轮样本、logits 与 greedy token 闸门 |
| `run_cuda_paged_prefill_ab.sh` | 比较分页逐 Token 与 Tensor Prefill | 两轮独立运行、跨页写入、logits 与 greedy token 闸门 |
| `run_cuda_batch_ab.sh` | 比较双请求分页顺序 Decode 与 Tensor Batch | 两轮独立运行、逐请求 logits 与 greedy token 闸门 |
| `run_cuda_paged_attention_ab.sh` | 比较分页三阶段与融合在线 Attention | 交替执行顺序、两轮样本、logits 与 greedy token 闸门 |

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

有 NVIDIA GPU、CUDA Toolkit、Nsight Systems 和本地 TinyLlama 时，可以生成不覆盖仓库结果的当前 CUDA kernel profile：

```bash
bash code/chapter09_performance/scripts/run_cuda_profile.sh
```

脚本固定 `Hello TinyLlama` 和四个 greedy token，保存环境、程序输出、kernel 汇总与设备内存活动 CSV。`.nsys-rep` 只在临时目录中生成并在结束后删除。该结果用于决定下一项融合优化，不替代端到端 A/B benchmark。

仓库保留同一 RTX 5070 Ti Laptop / CUDA 12.8 workload 的优化前后结果。EXP-GPU-02 位于 `results/profile/cuda/rtx5070ti-laptop-cuda12.8/`：Q4_K/Q6_K 反量化合计占 kernel 时间 `77.8%`，三阶段 GQA 约占 `0.7%`。EXP-GPU-03 位于相邻的 `rtx5070ti-laptop-cuda12.8-fused-gemv/`：Q4_K/Q6_K 融合 kernel 合计约 `134.1 ms`，旧反量化加主要 cuBLAS GEMV 约 `274.6 ms`，下降约 `51%`，golden token 保持一致。

脚本默认使用 EXP-GPU-03 和当前融合实现；需要显式指定输出身份时可运行：

```bash
CHAPTER09_CUDA_EXPERIMENT_ID=EXP-GPU-03 \
CHAPTER09_CUDA_OPTIMIZATION=fused-q4k-q6k-gemv-v1 \
bash code/chapter09_performance/scripts/run_cuda_profile.sh
```

融合量化 GEMV 仍占优化后 kernel 时间 `95%` 以上，说明下一轮应继续优化该 kernel 的内存访问与归约，而不是把百分比下降误解为量化执行已经完成。旧 EXP-GPU-02 对应融合前代码，只能从归档附件或历史提交复现。该结论只适用于固定的短提示词、四词元 Decode workload；长 Prefill 和长上下文需要单独测量。

Attention 微基准只能说明 kernel 本身是否值得进入候选。要决定 Runtime 默认路径，还需运行真实端到端 A/B：

```bash
bash code/chapter09_performance/scripts/run_cuda_attention_ab.sh
```

脚本显式传入 `--attention-path compare`，由可执行程序把字符串转换为 `CudaAttentionMode`，并在同一 workload 下运行两条受维护路径。也可以直接运行构建产物并指定 `three-stage` 或 `fused-online` 采集单路径 profile。该参数切换实现；实验 ID 和结果目录只记录证据，不参与实现选择。

EXP-GPU-04 固定 32-token Tensor Prefill、一次 Decode、2 次预热、15 次测量和两轮独立运行。融合路径的 logits 与 greedy token 均通过，但两轮中位数的快慢方向相反：第一轮 Prefill/Decode 分别回退约 `9.8%/4.7%`，第二轮分别改善约 `8.0%/14.5%`。该证据不支持切换默认实现，因此三阶段路径继续作为 Runtime 默认值，融合路径保留用于教学、Profiling 和后续调优。

分页 Tensor Prefill 使用独立 A/B 入口：

```bash
bash code/chapter09_performance/scripts/run_cuda_paged_prefill_ab.sh
```

EXP-GPU-05 固定同一分页 payload、BlockTable、32-token prompt 和三阶段 Attention，只改变“逐 Token 调用 Decoder”与“一次 Tensor Prefill”。两轮中位数从 `701.0/692.3 ms` 降至 `54.2/54.2 ms`，logits 最大误差为 `4.82e-05`，支持保留分页 Tensor Prefill。计时不含模型加载、Context 创建和请求绑定，结论不外推到其他 prompt 长度、并发批量或 GPU。

双请求 Tensor Batch 使用独立 A/B 入口：

```bash
bash code/chapter09_performance/scripts/run_cuda_batch_ab.sh
```

EXP-GPU-06 固定两个不同长度 prompt、分页 K/V 和一次 Decode，只改变“两个单请求依次执行”与“一次 Tensor Batch”。初版批量 Q4_K/Q6_K Linear 的两轮 batch/顺序中位数比例为 `0.922/0.919`，逐请求 logits 与 greedy token 一致。

EXP-GPU-08 使用相同 workload，进一步让最多 4 个 batch 行共享一次 packed 权重读取与解码，并把 block 归约改为 Warp shuffle 加 8 个 Warp 部分和。两轮比例降为 `0.466/0.612`，logits 最大误差为 `0`；Nsight Systems 中 Q4_K/Q6_K kernel 仍占 GPU kernel 时间约 `96.5%`。跨实验墙钟会受到 GPU 时钟影响，因此这些数字支持保留实现和继续优化 GEMV，不用于宣称固定倍数加速。计时不含模型加载、Context 创建、请求绑定和 Prefill，结论只覆盖当前双请求小批量 Decode。

取得 NCU 权限后，EXP-GPU-09～11 依次检查元数据广播、shared payload 和显式 FMA。按相同时钟下的 `Elapsed Cycles` 比较，三个候选均回退，因此只保存 profile 证据，不进入默认实现。EXP-GPU-12 将手写 FP16 位解码替换为 CUDA 原生 half 转换，Q4_K kernel 从 `140,093` 降至 `99,725` cycles，约减少 `28.8%`；真实双请求的顺序 Decode 两轮中位数为 `35.12/30.54 ms`，Tensor Batch 为 `19.80/17.02 ms`，logits 最大误差为 `0`。该实现通过 kernel 与端到端双重门禁，予以保留。

继续扩大 batch 前，先运行专项矩阵：

```bash
bash code/chapter09_performance/scripts/run_cuda_quantized_gemv_matrix.sh
```

EXP-GPU-13 固定 Q4_K/Q6_K、`2048 x 2048` 权重和 `batch=1/2/4/8`，输出相对重复单行调用的加速比，并让每个 batch 行与单行路径比较。Q4_K 的 batch 2/4/8 加速约为 `2.00x/3.98x/4.07x`，Q6_K 约为 `2.00x/4.00x/4.15x`；batch=8 时间约为 batch=4 的两倍，说明固定 4 行 tile 已把单次权重解码复用上限暴露出来。需要 NCU 定位某个点时，直接运行构建产物的 `--batch`、`--format` 和 `--iterations` 参数；这些参数选择 workload，不代表切换实现。

分页融合 Attention 使用独立 A/B 入口：

```bash
bash code/chapter09_performance/scripts/run_cuda_paged_attention_ab.sh
```

EXP-GPU-07 固定 32-token 分页 Tensor Prefill 和一次 Decode。第一轮融合 Prefill/Decode 分别回退约 `5.2%/21.7%`，第二轮分别改善约 `12.7%/5.6%`；两轮方向不一致，因此继续使用三阶段默认路径。融合 kernel 保留为在线 Softmax 与页表寻址实验，后续需用 Nsight 继续定位波动和物理块访问成本。

需要解释端到端结果时，启用同一脚本的 Nsight Systems 附件：

```bash
CHAPTER09_CUDA_PAGED_ATTENTION_PROFILE=1 \
bash code/chapter09_performance/scripts/run_cuda_paged_attention_ab.sh
```

归档 profile 中 Q4_K/Q6_K 融合 GEMV 占 GPU kernel 时间约 `99.2%`。分页融合 Attention 合计 `25.38 ms`，三阶段 score、Softmax 和 value 合计 `37.48 ms`，kernel 层约减少 `32.3%`；Attention 的绝对占比太小，端到端变化会被 Linear 与运行波动覆盖。采集前后图形时钟分别只有 `180/450 MHz`，也说明实验必须记录时钟、功耗、温度和利用率。下一轮性能工作应优先优化量化 GEMV，而不是继续用端到端墙钟反复调分页 Attention。

## 扩展入口

新增性能入口时先定义 workload 和正确性检查，再输出逐轮样本并转换为统一 evidence。CPU、CUDA、多模态和服务实验复用同一证据结构；平台专属 profiler 附件只负责归因，不替代 A/B 决策记录。
