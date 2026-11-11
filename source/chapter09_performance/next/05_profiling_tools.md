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

## 先用 nsys 找热点，再用 ncu 解释热点

Nsight Systems（`nsys`）观察系统时间线：CPU 线程、CUDA API、Kernel、内存复制和同步怎样交错。它适合回答“时间主要落在哪个 Kernel”“调用了多少次”“CPU 是否在等待 GPU”，不能解释单个 Kernel 内部为什么带宽低或 Warp 停顿。

Nsight Compute（`ncu`）会重放选中的 CUDA Kernel 并采集硬件计数器。它适合回答“显存带宽是否接近上限”“L2 命中率如何”“活跃 Warp 和占用率是否受寄存器或共享内存限制”。采集开销较大，也会改变程序时间，因此 `ncu` 输出用于解释 Kernel，不用于报告端到端延迟。

| 工具 | 先看什么 | 典型产物 | 不用于证明什么 |
| --- | --- | --- | --- |
| `nsys` | Kernel 总时间、实例数、CUDA API 和同步 | `.nsys-rep`、Kernel 汇总 CSV | 某个 Kernel 的硬件瓶颈已经确定 |
| `ncu` | DRAM 吞吐、L2 命中、占用率、Warp stall | `.ncu-rep`、单 Kernel 指标 | 请求端到端一定变快 |

本教程先运行已固定 workload 的脚本，避免手工命令遗漏模型和正确性闸门：

```bash
CHAPTER09_CUDA_PAGED_ATTENTION_PROFILE=1 \
bash code/chapter09_performance/scripts/run_cuda_paged_attention_ab.sh
```

脚本使用 `nsys profile` 生成临时报告，再用 `nsys stats --report cuda_gpu_kern_sum` 导出 Kernel 总时间、实例数和分布。结果先判断热点是否值得继续；EXP-GPU-07 中量化 GEMV 占约 `99.2%`，采用批次权重复用和 Warp 归约后的 EXP-GPU-08 仍占约 `96.5%`。因此分页 Attention 即使局部加速，也不是当前端到端首要瓶颈。

定位到 GEMV 后，再用 NCU 比较同一 kernel 的 cycles、共享内存、寄存器和 L1/TEX 指标。需要解释原因时继续采集 Compute、Instruction、Memory、Scheduler 和 Warp State section，核对指令数量、活跃与可调度 Warp、主要停顿和缓存事务。EXP-GPU-14 中固定 4 行与自适应 tile 采用相同矩阵、相同 launch 次数和独立进程；自适应实现的指令数减少约 `21%`～`37%`，共享内存减少 `50%`～`75%`，但 L1 命中率基本不变。这组指标把原因定位为减少无效 tile 行工作，而不是缓存策略改变。EXP-GPU-15 随后用真实 Decode 得出 `KEEP`，EXP-GPU-16 再确认收益覆盖 batch 1/2、上下文 16/128 和 Decode 1/8 步；三层证据共同支持默认路径变更。

确定单个 Kernel 是热点后，再对最小可执行 workload 使用 `ncu`。下面的命令只展示调用结构；Kernel 名称和可用指标会随 CUDA、GPU 架构与 Nsight Compute 版本变化，运行前先用 `ncu --query-metrics` 核对本机支持项。

```bash
ncu --set basic \
  --kernel-name regex:quantized_gemv_kernel \
  --launch-count 1 \
  <benchmark-executable> <model.gguf>
```

若出现 `ERR_NVGPUCTRPERM`，表示当前用户没有读取 NVIDIA GPU 性能计数器的权限。此时仍可使用 `nsys` 判断 kernel 时间占比，但 L2/DRAM 吞吐、占用率和 Warp stall 需要管理员开放 profiling counters 后再采集；不要用缺失的计数器推断具体硬件原因。

EXP-GPU-09～12 展示了 NCU 指标怎样改变优化方向。元数据广播和 shared payload 看似减少全局读取，却增加了 shuffle 或 barrier，最终 cycles 回退；原生 half 转换直接删除手写规格化分支，使 Q4_K kernel cycles 下降约 `28.8%`。判断时同时读取 `Elapsed Cycles` 与 `SM Frequency`，避免把动态时钟变化误判为代码收益。

分析完成后必须回到相同 workload 的端到端 A/B：`nsys` 决定热点优先级，`ncu` 解释硬件原因，墙钟样本和正确性闸门决定实现是否保留。

## 记录 GPU 运行状态

消费级 GPU 会随电源模式、温度和其他进程改变时钟。实验至少记录 GPU 名称、驱动、CUDA Toolkit、图形/显存时钟、功耗、温度和利用率；采集前后状态差异明显时，不把两轮墙钟结果直接归因于代码变化。`nvidia-smi` 快照只能描述采样时刻，严格实验还需要固定电源环境并持续采样。

## 验证阶段计时入口

有 TinyLlama GGUF 时，运行：

```bash
bash code/chapter09_performance/scripts/run_real_evidence.sh profile
```

脚本调用 <a href="https://github.com/datawhalechina/cpp-llm-engine/blob/main/code/engine/scripts/run_real_decoder_profile.sh"><code>run_real_decoder_profile.sh</code></a>，默认重复三轮，并在结果目录的 `profile/tinyllama-decoder-profile.txt` 保存 Decoder 阶段证据。文件先记录 experiment、workload、模型、CPU、编译器和构建类型，再记录正确性与阶段平均耗时；这些身份字段缺失时脚本直接失败。当前三轮归档结果见 <a href="https://github.com/datawhalechina/cpp-llm-engine/blob/main/code/chapter09_performance/results/profile/tinyllama-decoder-scalar.txt"><code>tinyllama-decoder-scalar.txt</code></a>。
