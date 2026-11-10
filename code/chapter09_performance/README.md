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
| `run_cuda_adaptive_gemv_ab.sh` | 比较固定 4 行与自适应 1/2/4 行 tile | 微基准、CPU reference、NCU 原始 CSV 与原因摘要 |
| `run_cuda_adaptive_gemv_runtime_ab.sh` | 在真实 TinyLlama Decode 中复核自适应 tile | 两轮交错执行、顺序路径保护指标与自动 `KEEP/REJECT/INCONCLUSIVE` 决策 |
| `run_cuda_adaptive_gemv_runtime_matrix.sh` | 扩展 batch、上下文长度和 Decode 步数 | 一次模型加载、两轮反向路径顺序、逐单元 logits 闸门与 JSON 摘要 |
| `run_cuda_paged_attention_ab.sh` | 比较分页三阶段与融合在线 Attention | 交替执行顺序、两轮样本、logits 与 greedy token 闸门 |
| `run_qwen15_moe_layer_cache.sh` | 比较 Qwen1.5-MoE 逐层读取与全层 LRU 缓存 | 反向执行顺序、生成词元、缓存命中与常驻字节数 |
| `run_qwen15_moe_grouped_batch_ab.sh` | 比较 MoE 专家逐行与分组 batch Linear | 同一权重、反向执行顺序、路由与逐元素误差闸门 |
| `run_qwen15_moe_batch_runtime_ab.sh` | 在完整 24 层 Decode 中复核 MoE 分组执行 | 双请求、反向执行顺序、logits 与贪心词元闸门 |
| `run_qwen15_moe_batch_runtime_matrix.sh` | 检查 MoE 分组执行的 batch 适用范围 | 一次模型加载、batch 1/2/4、反向顺序与逐步 logits 闸门 |
| `run_qwen15_moe_scheduling_ab.sh` | 在 Chapter08 固定请求队列中复核分组 MoE | 四请求、三个连续调度步、完整生成序列与批量调用计数 |
| `run_qwen15_moe_service_ab.sh` | 在 Chapter10 动态 admission 中复核分组 MoE | 运行中加入请求、完整输出、服务指标与终态释放 |

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

### MoE 层权重缓存

Qwen1.5-MoE 的默认 CPU 路径逐层读取权重，控制常驻内存；同一请求继续 Decode 时会再次读取 24 层。下面的实验把缓存容量设为 24，比较关闭缓存和保留全部层权重：

```bash
CPP_LLM_QWEN15_MOE_CACHE_ORDER=uncached-first \
bash code/chapter09_performance/scripts/run_qwen15_moe_layer_cache.sh

CPP_LLM_QWEN15_MOE_CACHE_ORDER=cached-first \
bash code/chapter09_performance/scripts/run_qwen15_moe_layer_cache.sh
```

EXP-CPU-12 的两轮反向顺序实验都保持生成词元一致。全层缓存让一次 Decode 从约 `4.29`～`4.61 s` 降至 `2.33`～`2.42 s`，同时常驻约 `6.56 GiB` 权重；第一次 Prefill 仍需加载全部层，没有稳定收益。每种顺序当前只有一个样本，所以结果用于解释时间与内存交换，不作为发布吞吐基线，缓存默认关闭。

### MoE 分组执行

Qwen1.5-MoE 的多个请求可能命中同一专家。逐行基线为每个 token 重复执行 Gate、Up 和 Down；分组路径先按专家收集 token，再让组内输入共享 packed 权重读取和量化块解码：

```bash
bash code/chapter09_performance/scripts/run_qwen15_moe_grouped_batch_ab.sh
```

EXP-CPU-13 使用真实第一层、batch 2、1 次预热和 5 次配对测量，每轮反转两条路径的先后顺序。逐行与分组中位数分别为 `151.825 ms` 和 `90.697 ms`，比值为 `0.597`；路由一致，最大输出误差约 `6.1e-10`。计时区间只覆盖 Router、专家和共享专家，因此结果支持把分组执行作为默认 MoE 子层路径，不支持宣称完整 24 层 Runtime 获得同等比例加速。

局部结果通过后，再运行完整 Decoder 复核：

```bash
bash code/chapter09_performance/scripts/run_qwen15_moe_batch_runtime_ab.sh
```

EXP-CPU-14 固定双请求、一次 Decode 和 24 层权重缓存，Prefill 位于计时区间外。3 轮配对样本中，逐行与分组中位数分别为 `4564.28 ms` 和 `3267.10 ms`，比值为 `0.716`；最大 logits 误差约 `3.19e-5`，贪心词元一致。该实验支持完整 Runtime 默认使用分组执行，结论范围仍限于当前 CPU、batch 2、单步 Decode 与全层缓存 workload，不代表开放环服务吞吐。

默认路径确定后，用矩阵检查 batch 范围：

```bash
bash code/chapter09_performance/scripts/run_qwen15_moe_batch_runtime_matrix.sh
```

EXP-CPU-15 在一次模型加载中依次测量 batch 1/2/4，每个单元执行两轮反向顺序配对。分组/逐行中位数比例分别为 `0.907/0.688/0.581`，所有单元最大 logits 误差约 `3.19e-5` 且生成词元一致。batch=1 的差异来自统一 batch backend 与单行兼容路径的实现差别；batch 增大后，同一专家组内共享权重解码的机会增加。两轮样本用于确认趋势与正确性，尚不足以作为稳定服务吞吐基线。

EXP-CPU-16 进一步固定 batch 1/2，对比一步和四步 Decode。batch 1 的分组/逐行比例为 `0.880/0.880`，batch 2 为 `0.668/0.699`；四步最大 logits 误差为 `5.72e-5`，生成词元仍一致。当前四步范围没有出现收益消失或离散输出漂移，但误差闸门仍需随更长序列继续观察。复现实验时显式登记新的实验和 workload 身份：

```bash
CPP_LLM_QWEN15_MOE_MATRIX_EXPERIMENT_ID=EXP-CPU-16 \
CPP_LLM_QWEN15_MOE_MATRIX_WORKLOAD_ID=WL-CPU-QWEN15-MOE-DECODE-STEP-MATRIX-V1 \
CPP_LLM_QWEN15_MOE_MATRIX_BATCHES=1,2 \
CPP_LLM_QWEN15_MOE_MATRIX_DECODE_STEPS=1,4 \
bash code/chapter09_performance/scripts/run_qwen15_moe_batch_runtime_matrix.sh
```

### MoE 调度层复核

Runtime 矩阵只直接调用 Factory。要确认 Chapter08 的计划、批量调用和请求提交不会抵消收益，运行固定请求队列实验：

```bash
bash code/chapter09_performance/scripts/run_qwen15_moe_scheduling_ab.sh
```

EXP-CPU-17 为四个请求各生成四个词元，Prefill 位于计时区间外，随后计时三个连续调度步。逐行与分组中位数分别为 `29.261 s` 和 `17.019 s`，比例为 `0.582`；四个请求的完整生成序列、三次模型批量调用和十二个批量请求推进均一致。该实验验证固定队列的持续批量执行，不包含运行中动态加入请求，因此还不能代表完整 continuous batching 到达过程或 Chapter10 服务吞吐。

### MoE 服务层复核

Chapter10 实验先让两个请求完成 Prefill，再在它们 Decode 期间加入两个新请求，覆盖动态 admission、混合阶段调度和统一终态释放：

```bash
bash code/chapter09_performance/scripts/run_qwen15_moe_service_ab.sh
```

EXP-CPU-18 的逐行与分组中位数分别为 `32.334 s` 和 `22.347 s`，比例为 `0.691`。两轮反向顺序配对都保持四请求完整输出一致，accepted、completed、released 和批量执行形状一致。计时不包含最初两个请求的 Prefill，但包含新增请求 admission、后续 Prefill/Decode 调度、模型执行和终态释放；固定四请求的闭环实验仍不等于开放环到达率或真实 HTTP 并发吞吐。

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

EXP-GPU-14 比较固定 4 行 tile 与按 batch=1/2/其他选择 1/2/4 行的自适应实现。两轮微基准、8 份 NCU CSV 和 CPU reference 均通过；详细 NCU 显示执行指令减少约 `21%`～`37%`、静态共享内存减少 `50%`～`75%`，Q4_K batch 1/2 的 cycles 约下降 `17%/19%`，Q6_K 约下降 `33%/30%`。L1 命中率基本不变，占用率只提高约 2.5～3.1 个百分点，因此主因是模板不再展开不存在的 tile 行，而不是缓存命中率改善。EXP-GPU-15 再进入真实 TinyLlama 分页 Decode：Tensor Batch 两轮分别改善约 `16.3%/19.1%`，顺序 Decode 没有稳定回退，logits 最大误差为 `0`。代码据此默认选择 `Adaptive`，并保留 `FixedTile4` 复现实验基线。

EXP-GPU-16 检查默认策略的适用范围。最终矩阵对 `batch=1/2`、上下文 `16/128` 和 Decode `1/8` 步执行两轮反向顺序实验，每轮预热 2 次、测量 5 次。batch 1 的 Tensor Batch 延迟降低约 `18.4%`～`23.1%`，batch 2 降低约 `12.9%`～`14.7%`，所有单元的 logits 最大误差为 `0`。前置探索矩阵中的 batch 4/8 继续使用 4 行 kernel，长上下文下变化在约 `±0.7%` 内，作为噪声控制而不计入优化收益。

两个脚本现在分开报告三种状态：`execution_status` 说明命令是否完整执行，`correctness` 说明数值闸门是否通过，`decision` 才说明候选应当 `KEEP`、`REJECT` 还是证据不足的 `INCONCLUSIVE`。真实 Decode 决策要求两轮 Tensor Batch 均至少改善 `5%`、每轮顺序 Decode 回退不超过 `5%`，并且 logits 最大误差不超过 `1e-3`。NCU 摘要保存 cycles、指令、Warp 调度、占用率和内存层级指标；缺失指标保持缺失，不用其他计数器推测。

```bash
CUDA_HOME=/usr/local/cuda-12.8 \
bash code/chapter09_performance/scripts/run_cuda_adaptive_gemv_ab.sh

CUDA_HOME=/usr/local/cuda-12.8 \
bash code/chapter09_performance/scripts/run_cuda_adaptive_gemv_runtime_ab.sh
```

默认路径确定后，再用独立矩阵检查适用范围：

```bash
CUDA_HOME=/usr/local/cuda-12.8 \
bash code/chapter09_performance/scripts/run_cuda_adaptive_gemv_runtime_matrix.sh
```

程序只加载一次模型，但每个测量样本重新创建并 Prefill 请求；计时区间只包含 Decode。两轮分别采用 `fixed-first` 与 `adaptive-first`，`summary.json` 按同一单元合并结果。环境变量可以调整 batch、上下文、Decode 步数、预热和采样次数；扩大到上下文 `512` 或 Decode `32` 步时应登记为新 workload，不与 EXP-GPU-16 直接合并。

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
