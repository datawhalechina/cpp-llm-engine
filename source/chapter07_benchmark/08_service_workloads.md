# 07.8 服务负载评测

多请求 benchmark 必须先说明执行方式。当前仓库同时具有教学 Transformer Tensor batch、真实模型 scalar Linear batch、真实模型串行请求和 CPU 请求级并发。真实 batch 共享权重，在每层把多个 Context 的激活合并到 Q/K/V、输出投影、FFN 和 LM Head；Attention 与连续 KV 仍按请求处理。

## 当前可复现的五种负载

| 负载 | 模型与执行方式 | 能回答什么 | 不能回答什么 |
| --- | --- | --- | --- |
| Chapter07 Round-robin | 教学 Transformer，逐请求推进 1/2/4 请求 | 逐请求调用基线 | 真实模型吞吐 |
| Chapter07 Tensor batch | 教学 Transformer，连续 ragged batch 一次推进多个请求 | batch 数据契约、golden 和模型调用次数 | 真实 Decoder、Paged KV 和 GPU 收益 |
| TinyLlama scalar Linear batch | 22 层内合并多个 Context 的量化 Linear | 顺序/batch logits、golden 和双请求 decode 墙钟 | 专用 AVX2 batch、Paged KV 或其他机器表现 |
| TinyLlama sequential | 真实 22 层模型，两个独立 Context 在一个线程轮转 | 串行服务基线、TTFT 与阶段时间 | CPU 请求并行收益 |
| TinyLlama parallel-requests | 两个 CPU worker 并行运行独立 Executor/Context | 请求级并发的墙钟与吞吐变化 | 单次 Tensor batch、PagedAttention |

真实 A/B 从仓库根目录运行：

```bash
bash code/engine/scripts/run_real_parallel_requests_ab.sh
```

脚本固定 TinyLlama Q4_K_M、optimized AVX2、双请求、每请求 4 个生成 token、warmup 1 和测量 3 轮，生成 sequential 与 parallel-requests 两份 JSON。两种模式必须同时通过 golden token、Context/cache 隔离和终态资源释放。

## 本次请求级并发证据

当前归档环境为 AMD Ryzen 9 9955HX、GCC 13.1.0、Release。请求级并发把两个真实 Decoder Context 放到两个 CPU worker；累计模型计算时间可以重叠，所以 JSON 同时记录墙钟 `dispatch` 和 `overlapped_compute`。

| 指标 | Sequential | Parallel requests | 变化 |
| --- | ---: | ---: | ---: |
| Dispatch mean | `1528.87 ms` | `1064.34 ms` | `-30.38%` |
| Throughput mean | `5.233 token/s` | `7.517 token/s` | `+43.64%` |
| Overlapped compute | `0 ms` | `531.60 ms` | 出现真实重叠 |
| Correctness | PASS | PASS | 可比较 |

这组结果只支持“当前 CPU 和双请求短输出下，请求级并发产生了重叠并改善墙钟时间”。它不证明更多 worker 会继续线性加速，也不代表成熟框架的 Continuous Batching 或 Tensor batch。

## Scalar batch 配对结果

[`EXP-CPU-10`](../reference/experiment_assets.md#exp-cpu-10) 在 AMD Ryzen 9 9955HX、Release、双请求和每请求 3 次 decode 下执行一次预热与三轮配对测量。顺序路径均值为 `3604.68 ms`，scalar batch 均值为 `4220.34 ms`，配对加速比为 `0.8541x`；三轮 golden 与两种路径的 token 都一致。

该结果不支持把当前 scalar batch 作为性能优化启用。Chapter08 继续拆分 batch Linear 后发现：量化 kernel 占 batch profile 总时间 `99.8848%`，输入整理与输出拆分合计只有 `0.0184%`。因此当前证据指向多输出行 scalar 累加，而不是激活拼接/拆分。

## 从结果进入性能归因

`0.8541x` 说明“合并了计算”不等于“已经加速”。进入 [`Chapter08`](../chapter08_profiling/intro.md) 后，可用同一份 JSON 验证瓶颈位于 scalar batch kernel；下一轮改动应只替换专用多行 SIMD kernel，并保持 workload 与 correctness gate 不变。
