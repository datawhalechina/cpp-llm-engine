# Chapter 08：Profiling Evidence 性能证据

本目录不重新实现 Runtime，也不伪造硬件 trace。它保留重构前的结果文件、experiment/workload 身份和 correctness-first A/B 判定。当前学习说明见 [Chapter09 Performance](../../../../source/chapter09_performance/next/intro.md)。

## 本章资产总览

| 资产 | 作用 |
| --- | --- |
| `include/chapter08_profiling/profile_record.h` | benchmark 结果字段、解析和报告契约 |
| `include/chapter08_profiling/ab_compare.h` | 正确性优先的 A/B 接受规则 |
| `src/` | 结果解析、指标计算和差异判断 |
| `examples/main_profiling.cpp` | 报告和 A/B 命令行入口 |
| `tests/chapter08_profiling_test.cpp` | 解析、吞吐和回退阈值测试 |
| `scripts/select_multi_request_profile.mjs` | 从 Chapter 07 JSON 选择指定请求数并适配 ProfileRecord |
| `scripts/select_real_service_profile.mjs` | 把真实 TinyLlama ServiceCore JSON 适配为 ProfileRecord |
| `scripts/analyze_real_batch_profile.mjs` | 校验真实 scalar batch schema v2，并定位输入整理、kernel 与输出拆分耗时 |
| `scripts/run.sh` | 消费 Chapter 07 与归档真实服务基线，并执行轻量回归 |
| `scripts/run_real_service_ab.sh` | 重新采集真实服务 experiment，与归档 baseline 做同条件回归 A/B |
| `scripts/run_real_q6k_down_ab.sh` | 只切换 FFN down backend 的三轮 scalar/AVX2 A/B |
| `scripts/compare_q6k_down_profiles.mjs` | 校验 workload、正确性和计时闭合后生成 Q6_K A/B 判定 |
| `scripts/run_real_q4k_ffn_ab.sh` | 只切换 FFN gate/up backend 的三轮 scalar/AVX2 A/B |
| `scripts/compare_q4k_ffn_profiles.mjs` | 归档 Q4_K 实验的 `KEEP` 或 `REJECT` 判定 |
| `scripts/run_real_q4k_attention_ab.sh` | 只切换 QKV/out backend 的三轮 scalar/AVX2 A/B |
| `scripts/compare_q4k_attention_profiles.mjs` | 校验 Attention、FFN、LM Head 作用域后生成判定 |
| `scripts/run_real_optimized_policy_ab.sh` | 组合三个 AVX2 作用域并运行 Decoder/ServiceCore A/B |
| `scripts/compare_optimized_policy.mjs` | 校验组合 policy、正确性、环境和端到端收益 |
| `results/tinyllama_service_repeatability_experiment.json` | 当前工作树真实服务重复采集原始数据 |
| `results/tinyllama_service_repeatability_ab.txt` | baseline/experiment 报告和回归判定 |
| `results/tinyllama_decoder_phase_profile.txt` | 真实 Decoder 阶段及 FFN gate/up/activation/down/residual 三轮平均结果 |
| `results/tinyllama_q6k_down_*.txt` | Q6_K down scalar/AVX2 原始 profile 与 A/B 报告 |
| `results/tinyllama_q4k_ffn_*.txt` | Q4_K gate/up scalar/AVX2 原始 profile 与 A/B 报告 |
| `results/tinyllama_q4k_attention_*.txt` | Q4_K QKV/out scalar/AVX2 原始 profile 与 A/B 报告 |
| `results/tinyllama_optimized_*` | 组合 policy 的 Decoder、ServiceCore 原始结果与判定 |
| `results/tinyllama_real_batch_profile.txt` | `EXP-CPU-10` 的离线瓶颈摘要 |

## 运行与验证

从仓库根目录执行：

```bash
bash code/chapter08_profiling/scripts/run.sh
```

脚本默认选择 `request_count=4` 的 Chapter 07 JSON 场景，先生成阶段报告，再执行同构 baseline/experiment 的 A/B smoke。这个 A/B 只验证 JSON 适配、correctness 和判定逻辑，不代表优化收益；真实实验必须替换 experiment 输入，并保持 workload、硬件、编译选项和 correctness 条件一致。

默认脚本还会读取 `code/chapter09_serving/results/tinyllama_service_cpu_baseline.json`，报告真实模型的 prefill、decode、Runtime overhead、双请求 TTFT/完成时间差、权重 payload、已提交 KV、进程峰值 RSS 和瓶颈范围，但不会重新运行 638 MiB 模型。需要重新采集真实 experiment 时执行：

```bash
bash code/chapter08_profiling/scripts/run_real_service_ab.sh
```

该脚本默认使用归档 JSON 作为 baseline，以当前工作树重新构建并采集 `warmup=1/iterations=3` experiment。它验证环境、workload、correctness 和 5% 总延迟回退阈值，定位是回归/可重复性检查，不是单变量优化收益。

Q6_K down 单变量实验执行：

```bash
bash code/chapter08_profiling/scripts/run_real_q6k_down_ab.sh
```

脚本使用同一 Release 可执行文件、模型、prompt、token 数和三轮迭代，只把 FFN down 从 scalar fused 切换为 AVX2/FMA fused。只有 golden、计时闭合、down 延迟和 Decoder 总延迟全部通过才输出 `KEEP`。

Q4_K gate/up 使用同一套 workload 和正确性闸门：

```bash
bash code/chapter08_profiling/scripts/run_real_q4k_ffn_ab.sh
```

该脚本保持 Q6_K down 和其他 Q4_K projection 为 scalar，只切换 FFN gate/up。性能没有改善时仍归档 `REJECT`，只有正确性或可比性失败才中止。

Q4_K QKV/out 实验复用同一 policy 和正确性闸门：

```bash
bash code/chapter08_profiling/scripts/run_real_q4k_attention_ab.sh
```

该脚本保持 FFN 和 LM Head 为 scalar，只切换 Attention 的 Q、K、V 和 output projection。三轮结果中，两类 projection 合计下降 `80.7509%`，Decoder 总耗时下降 `17.3494%`，判定 `KEEP`。

三个作用域组合后的端到端实验执行：

```bash
bash code/chapter08_profiling/scripts/run_real_optimized_policy_ab.sh
```

脚本在同一 Release 构建中比较 scalar 与 `optimized-avx2`，同时检查 Decoder golden/profile、双请求 golden、Context/cache 隔离和资源释放。当前归档结果为 Decoder `-79.2998%`、ServiceCore dispatch `-78.4838%`、吞吐 `+364.7483%`，判定 `KEEP`；正常服务仍默认 scalar。
