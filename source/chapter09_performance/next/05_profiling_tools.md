# 09.5 按问题选择性能剖析工具

Profiling 应从低开销的总量观测开始，确认问题范围后再使用更细的工具。直接进入指令或 Kernel 级分析，容易在尚未固定 workload 时得到无法复现的热点。

| 顺序 | 问题 | CPU 常用方法 | CUDA 常用方法 |
| --- | --- | --- | --- |
| 1 | 回退是否真实存在 | 重复 Benchmark、阶段计时 | CUDA Event、重复 Benchmark |
| 2 | 时间落在哪个阶段 | `std::chrono`、范围计时 | NVTX、CUDA Event |
| 3 | 哪个函数或调用最重 | `perf stat`、`perf record` | Nsight Systems |
| 4 | 为什么这个热点慢 | cache miss、分支、SIMD 指标 | Nsight Compute 的带宽、占用率和指令指标 |

## 同步边界决定测到什么

CPU 计时通常在函数返回后完成；CUDA Kernel launch 是异步的，只测 launch 前后会得到提交开销，而不是 Kernel 执行时间。GPU 阶段测量需要 Event 或明确同步，端到端测量还要包括必要的数据搬运和调度成本。

| 目标 | 计时边界 |
| --- | --- |
| Kernel 本体 | 同一 stream 上的 CUDA Event |
| 一组 GPU 操作 | 范围开始/结束 Event 与一次必要同步 |
| 请求端到端 | Host 请求入口到结果可用 |
| 服务体验 | 网络入口到首词元、相邻词元和完成 |

## 验证阶段计时入口

有 TinyLlama GGUF 时，运行：

```bash
bash code/chapter09_performance/scripts/run_real_evidence.sh profile
```

脚本调用 <a href="../../_code/engine/scripts/run_real_decoder_profile.sh"><code>run_real_decoder_profile.sh</code></a>，默认重复三轮，并在结果目录的 `profile/tinyllama-decoder-profile.txt` 保存 Decoder 阶段证据。文件先记录 experiment、workload、模型、CPU、编译器和构建类型，再记录正确性与阶段平均耗时；这些身份字段缺失时脚本直接失败。当前三轮归档结果见 <a href="../../_code/chapter09_performance/results/profile/tinyllama-decoder-scalar.txt"><code>tinyllama-decoder-scalar.txt</code></a>。
