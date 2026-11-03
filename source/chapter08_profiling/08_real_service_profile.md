# 08.8 分析真实 TinyLlama 服务

本节从 Chapter09 的双请求 ServiceCore benchmark 进入真实 Decoder。所有结果先通过 golden token、Context/cache 隔离和终态资源释放，再用服务阶段计时判断优化对象位于模型计算还是服务编排。

## 从服务墙钟进入 Decoder

归档基线来自 AMD Ryzen 9 9955HX、GCC 13.1.0、Release 和 TinyLlama Q4_K_M。模型 prefill 与 decode 占 dispatch 的 `99.9979%`，Runtime/ServiceCore 开销为 `0.0021%`，因此下一步应拆解 Decoder，而不是先重写 admission 或队列。

请求级并发实验只切换 dispatch mode。双 CPU worker 使 dispatch 从 `1528.87 ms` 降到 `1064.34 ms`，吞吐从 `5.233` 提升到 `7.517 token/s`；`overlapped_compute=531.60 ms` 证明两个 Executor 的计时间隔发生重叠，但不等同于 Tensor batch 或 CPU 利用率证据。

Decoder 分段计时由 `real_tinyllama_decoder_profile` 显式开启。当前环境没有普通用户 `perf` 权限，因此这些结果属于代码内阶段证据，不替代符号采样和硬件计数器。

![真实 TinyLlama CPU Profiling 证据](../assets/diagrams/chapter08-real-service-profile.svg)

## 找到量化 Linear 热点

固定输入为 `Hello TinyLlama`、4 个生成 token 和三轮运行。阶段结果把主要时间定位到 FFN 与 Attention projection，而不是短 prompt 下的 Attention softmax。

FFN 内部的 Q6_K down、Q4_K up 和 Q4_K gate 分别占 `34.7888%`、`32.5832%` 和 `32.5780%`；SwiGLU、residual 与未归因时间合计约 `0.050%`。该证据支持先优化三组量化 Linear，不支持优先融合激活或残差。

| Decoder 子阶段 | 总时间占比 | 当前判断 |
| --- | ---: | --- |
| FFN | `76.2491%` | 第一优化范围，继续拆 gate/up/down Linear |
| QKV projection | `11.3654%` | 检查 Q4_K fused backend |
| Attention output projection | `9.0479%` | 与 QKV 复用量化 Linear 路径 |
| LM Head | `3.2634%` | 独立保留，不混入前三组实验 |
| Attention 本体 | `0.0219%` | 当前短 prompt 下不是主瓶颈 |

## 用单变量实验收敛结论

每项实验都使用相同模型、prompt、Release 构建和三轮配对测量，只切换表中指定 backend。原始结果、环境字段和适用范围集中在[实验与验证资产索引](../reference/experiment_assets.md)。

组合结果只证明这些 backend 在记录的 CPU 与固定 workload 下可以共同工作。默认服务仍使用 scalar，`optimized-avx2` 需要显式选择；其他 CPU、模型和长度必须重新通过能力检查、correctness 和 A/B。

| 实验 | 唯一变量 | 关键结果 | 复现入口 |
| --- | --- | --- | --- |
| `EXP-CPU-03` | 不改实现，复跑服务基线 | dispatch `+0.0922%`，未越过 5% 回归门 | `bash code/chapter08_profiling/scripts/run_real_service_ab.sh` |
| `EXP-CPU-05` | Q6_K FFN down：scalar → AVX2 | down `-41.5797%`，Decoder `-13.1518%`，`KEEP` | `bash code/chapter08_profiling/scripts/run_real_q6k_down_ab.sh` |
| `EXP-CPU-06` | Q4_K FFN gate/up：scalar → AVX2 | 两个投影约 `-82.8%`，Decoder `-40.6436%`，`KEEP` | `bash code/chapter08_profiling/scripts/run_real_q4k_ffn_ab.sh` |
| `EXP-CPU-07` | Q4_K Q/K/V/out：scalar → AVX2 | projection 合计 `-80.7509%`，Decoder `-17.3494%`，`KEEP` | `bash code/chapter08_profiling/scripts/run_real_q4k_attention_ab.sh` |
| `EXP-CPU-08` | 组合前三组已通过策略 | Decoder `-79.2998%`，ServiceCore `-78.4838%`，吞吐 `+364.7483%` | `bash code/chapter08_profiling/scripts/run_real_optimized_policy_ab.sh` |

## 定位独立 batch 路径的回退

`EXP-CPU-10` 不经过 ServiceCore，而是比较两个真实 Context 的顺序 decode 与 scalar Linear batch。schema v2 在每轮结果中增加输入整理、量化 kernel 和输出拆分计时，并继续使用相同 golden token 作为正确性闸门。

三轮结果中，顺序路径为 `3604.68 ms`，batch 为 `4220.34 ms`，配对加速比为 `0.8541x`。batch 内部的 465 次量化 Linear 调用耗时 `4215.32 ms`，占 profile 总时间 `99.8848%`；输入整理和输出拆分合计只有 `0.78 ms`。因此下一项单变量实验应替换多行 scalar kernel，而不是先改激活容器或 ServiceCore 调度。

| 观察对象 | 三轮均值 | 当前结论 |
| --- | ---: | --- |
| 输入整理 | `0.21 ms` | 不是当前主要回退来源 |
| scalar batch kernel | `4215.32 ms` | 首要优化对象 |
| 输出拆分 | `0.56 ms` | 不是当前主要回退来源 |
| 配对加速比 | `0.8541x` | 只保留正确性路径 |

## 核对真实内存口径

同一服务实验还记录共享 Model 的权重 payload、两个 Context 在释放前已经提交的 K/V，以及进程峰值 RSS。请求 A 的 prompt 与生成历史共提交 9 个 token，占 `405504 B`；请求 B 提交 4 个 token，占 `180224 B`。两者之和 `585728 B` 通过 `层数 × KV 头数 × 头维度 × K/V 两份 × float 字节数 × token 数` 反算校验。

峰值 RSS 包含模型之外的进程开销，不能解释为“权重加 KV”。这组服务证据测量的仍是 `RealLlamaCpuContext` 连续 K/V，不是 Chapter06 独立验证的 CPU 分页 payload。

| 观察对象 | `EXP-CPU-02` | `EXP-CPU-03` |
| --- | ---: | ---: |
| 模型权重 payload | `667078656 B` | `667078656 B` |
| 双请求已提交 KV | `585728 B` | `585728 B` |
| 进程峰值 RSS | `652.82 MiB` | `652.32 MiB` |

## 验证与衔接

复现时先运行 policy 单测和真实 golden，再比较 QKV、FFN、Decoder 与 ServiceCore 指标。真实 batch 的归档 JSON 可由 `analyze_real_batch_profile.mjs` 离线校验；重新采集则运行 `run_real_batch_benchmark.sh`。具备系统 profiler 权限时，在相同 workload 上补充符号或硬件证据；证据能支持到哪一级结论，见 [`08.9 Profiling 证据层级`](./09_profiling_evidence_hierarchy.md)。
