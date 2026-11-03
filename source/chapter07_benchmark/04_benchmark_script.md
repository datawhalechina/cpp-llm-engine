# 07.4 用脚本复现基准结果

手工执行若干程序容易遗漏构建参数、预热次数或输出路径。本节阅读仓库已有的 `run.sh`，观察一次基准怎样从隔离构建走到机器可读结果和 CTest。

## 默认脚本的执行顺序

<a href="../_code/chapter07_benchmark/scripts/run.sh"><code>scripts/run.sh</code></a> 依次完成以下步骤：

| 顺序 | 程序 | 产物 |
| --- | --- | --- |
| 1 | CMake 配置与构建 | 临时构建目录中的可执行文件 |
| 2 | `chapter07_benchmark_demo` | 固定阶段、事件和 trace 格式样例 |
| 3 | `chapter07_runtime_benchmark` | Chapter06 Runtime 的单请求实测 |
| 4 | `chapter07_multi_request_benchmark` | 两种执行模式、`1/2/4` 请求实测和 JSON 文件 |
| 5 | CTest | Chapter06 与 Chapter07 回归结果 |

默认使用 Release 构建、一次预热和三次测量。可用 `CHAPTER07_BUILD_TYPE`、`CHAPTER07_WARMUP`、`CHAPTER07_ITERATIONS` 和 `CHAPTER07_RESULT_DIR` 改变构建类型、次数与输出目录；比较两组实现时，两边必须使用相同设置。

## 结果文件与可复现条件

默认结果写入 `code/chapter07_benchmark/results/chapter07_cpu_baseline.json`。为了避免覆盖仓库基线，可以把实验写到临时目录：

```bash
CHAPTER07_RESULT_DIR=/tmp/chapter07-results \
CHAPTER07_WARMUP=2 \
CHAPTER07_ITERATIONS=10 \
bash code/chapter07_benchmark/scripts/run.sh
```

固定脚本仍不能自动消除系统噪声。schema v2 会记录 CPU、编译器、构建类型、每轮样本和统计值；正式比较仍要补充 Git 提交、线程设置、频率策略和后台负载，不能从耗时数字反推这些条件。

结果顶部还有两个身份字段。`experiment_id` 把结果连接到实验资产索引，例如教学多请求基线使用 `EXP-CPU-01`；`workload_id` 固定模型、输入、请求数和测量边界，Chapter08 选择具体场景后会继续附加请求数。Round-robin 与 Tensor batch 是同一 workload 下被比较的执行方式，不写入 workload ID。

真实 TinyLlama 服务使用另一条入口：

```bash
bash code/engine/scripts/run_real_serving_benchmark.sh
```

该脚本依赖外部 GGUF，记录模型加载和服务阶段数据，不进入默认快速回归。

真实 sequential/scalar Linear batch 使用配对入口：

```bash
bash code/engine/scripts/run_real_batch_benchmark.sh
```

脚本固定两个 prompt、每请求 4 个输出 token，只计量三个 decode 调用，排除模型加载、Context 创建和 prefill。每轮同时测量 sequential 与 batch，并交替执行顺序；结果以 `EXP-CPU-10 / WL-CPU-REAL-BATCH-V1` 写入 `code/chapter07_benchmark/results/tinyllama_real_batch_scalar.json`。该结果只回答当前 scalar batch 在固定双请求负载下是否改善 decode 墙钟，不代表 AVX2、GPU 或服务开放环吞吐。

## 验证与衔接

在临时目录运行两次，确认 `experiment_id` 和 `workload_id` 未变化、六个场景均为 `correctness=true`，每种执行模式都包含 `1/2/4` 请求，且 `samples` 数量等于 `iterations`。下一篇 [`07.5 测量稳定性`](./05_optimization_tips.md) 解释如何使用统计范围判断耗时波动是否允许比较。
