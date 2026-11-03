# 07.3 定义延迟与吞吐指标

同一组耗时只有在起止点、输入规模和聚合方式一致时才可比较。本节使用 Chapter07 结果文件中的字段，连接预填充、逐词元解码和完整请求三个观察范围。

## 指标与测量区间

| 指标 | 当前起止点 | 单位 | 阅读重点 |
| --- | --- | --- | --- |
| `prefill_ms` | 调用所有请求的 `Runtime::prefill` 前后 | ms | prompt 处理成本；当前请求按顺序执行 |
| `decode_ms` | Round-robin 调度开始到全部请求结束 | ms | 多请求逐步推进的总耗时 |
| `total_ms` | `prefill_ms + decode_ms` | ms | 教学程序中的模型执行总时间 |
| `ttft_ms` | prefill 开始到第一次 decode 回调 | ms | 当前教学定义，不含网络与排队 |
| `throughput_tokens_per_second` | 生成词元总数除以 `total_ms` | token/s | 固定负载下的整体生成速率 |

真实 HTTP 服务中的 TTFT 还会包含排队、序列化和 transport 时间，因此不能直接和当前教学 Runtime 的 `ttft_ms` 横向比较。跨工具比较前，先核对公式、计时边界和分母。

## 工作负载决定数字含义

Chapter07 默认比较 `1/2/4` 个请求，每个请求有 4 个 prompt token，并生成 4 个 token。每组负载分别运行 Round-robin 和教学 Tensor batch：前者逐请求调用模型，后者把请求合并为一次 `Transformer::prefill_batch()` 或 `decode_batch()`。两者共享输入、生成上限和正确性检查。

结果至少要同时保留以下条件：

| 条件 | 当前结果字段或入口 |
| --- | --- |
| 请求规模 | `request_count`、`prompt_tokens`、`max_new_tokens` |
| 执行方式 | `execution_mode`、`model_invocations`、`tensor_batch_steps` |
| 重复方式 | 顶层 `warmup`、`iterations` |
| 执行环境 | `environment.cpu_model`、`hardware_concurrency`、`compiler`、`build_type` |
| 正确性 | `generated_tokens`、`completed_requests`、`correctness` |
| 时间与速率 | prefill、decode、total、TTFT、throughput |
| 波动范围 | 每轮 `samples` 和 `statistics` 中的 mean/median/min/max |

结果 schema v2 已记录 CPU 型号、逻辑并发数、编译器和构建类型。它仍未记录 CPU 频率策略、内存配置、操作系统与后台负载，正式实验需要把这些条件随命令补充。真实服务 benchmark 的阶段和字段不同，两者不要混为同一种结果格式。

## 验证与衔接

运行默认脚本后查看结果：

```bash
bash code/chapter07_benchmark/scripts/run.sh
sed -n '1,220p' code/chapter07_benchmark/results/chapter07_cpu_baseline.json
```

确认每个场景都同时包含负载、正确性和性能字段。下一篇 [`07.4 可重复执行脚本`](./04_benchmark_script.md) 将说明脚本怎样固定这些条件并写出结果。
