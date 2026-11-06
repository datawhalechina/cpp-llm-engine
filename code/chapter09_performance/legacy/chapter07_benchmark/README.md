# Chapter 07：Benchmark 基准测试与证据

本目录把正确性测试、阶段耗时、事件 trace、热点摘要和 JSON/Chrome trace 组织成可重复的历史 benchmark 入口。当前学习说明见 [Chapter09 Performance](../../../../source/chapter09_performance/next/intro.md)。

## 本章资产总览

| 资产 | 作用 |
| --- | --- |
| `src/benchmark.cpp` | benchmark 计时和阶段记录 |
| `src/test_helpers.cpp` | 固定输入和测试辅助 |
| `tests/chapter07_benchmark_test.cpp` | correctness 和 benchmark 输出测试 |
| `tests/check_multi_request_result.cmake` | 执行两轮测量并检查 schema v2 证据字段 |
| `examples/main_benchmark.cpp` | 生成阶段、事件和 trace 示例 |
| `examples/runtime_benchmark.cpp` | Chapter 06 CPU runtime 实测 benchmark |
| `examples/multi_request_benchmark.cpp` | 教学 Transformer 的 `1/2/4` 请求 Round-robin/Tensor batch 对照与 JSON 输出 |
| `../engine/scripts/run_real_parallel_requests_ab.sh` | 真实 TinyLlama sequential/parallel-request A/B |
| `scripts/run.sh` | 隔离构建、CTest 和 benchmark 入口 |
| `results/chapter07_cpu_baseline.json` | 最近一次可复查的 CPU 多请求基线结果 |
| Chapter 08 | 消费本章 trace 和 hotspot 证据进行 profiling |

## 运行与验证

从仓库根目录执行：

```bash
bash code/chapter07_benchmark/scripts/run.sh
```

`runtime_benchmark` 输出的是 Chapter 06 CPU runtime 的实测结果；`main_benchmark` 中的固定 trace 只用于演示结果格式。比较结果时固定输入、设备、编译选项和脚本版本；benchmark 输出用于支持决策，不替代正确性测试。

`multi_request_benchmark` 对同一个教学 CPU Runtime 分别使用 Round-robin 和 `TensorBatchScheduler`，每种模式测量 1、2、4 个请求。脚本默认使用 Release、warmup=1、iterations=3，并将结果写入 `code/chapter07_benchmark/results/chapter07_cpu_baseline.json`。schema v2 保存 `EXP-CPU-01`、稳定 workload ID、执行模式、模型调用次数、环境、逐轮样本和统计值；`correctness`、请求完成数和生成 token 数仍是硬性验收条件。

可通过 `CHAPTER07_BUILD_TYPE`、`CHAPTER07_WARMUP`、`CHAPTER07_ITERATIONS` 和 `CHAPTER07_RESULT_DIR` 覆盖默认条件。比较两组结果时必须使用相同设置。

真实请求级并发使用 `bash code/engine/scripts/run_real_parallel_requests_ab.sh`。两个 CPU worker 分别执行独立 Executor/Context；该结果不代表单次 Decoder Tensor batch。
