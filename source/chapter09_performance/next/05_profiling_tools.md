# 09.5 按问题选择性能剖析工具

Profiling 应从低开销的总量观测开始，确认问题范围后再使用更细的工具。直接进入指令或 Kernel 级分析，容易在尚未固定负载时得到无法复现的热点。

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

两者分工可以概括为：`nsys` 选择值得分析的计算核，`ncu` 解释这个计算核受计算、访存还是调度限制。先用 `ncu` 随机挑选计算核，容易花费很长时间采集一个对端到端延迟影响很小的局部。

## 检查本机工具

Nsight Systems 和 Nsight Compute 随 CUDA Toolkit 或 NVIDIA Nsight 工具安装，具体路径取决于安装方式。本教程通过 `CUDA_HOME` 定位 CUDA 工具，先检查版本和设备状态：

```bash
export CUDA_HOME=/usr/local/cuda-12.8

"${CUDA_HOME}/bin/nsys" --version
"${CUDA_HOME}/bin/ncu" --version
nvidia-smi
```

版本检查失败时，先确认 `CUDA_HOME` 是否指向实际安装目录。只有 `ncu` 报 `ERR_NVGPUCTRPERM` 时，才需要检查 GPU 性能计数器权限；`nsys` 的 CUDA 时间线通常不依赖这些硬件计数器。

## 用 nsys 建立时间分布

TinyLlama CUDA 生成入口会完成 Release 构建、固定四词元生成、正确性检查和报告导出：

```bash
CUDA_HOME=/usr/local/cuda-12.8 \
CHAPTER09_CUDA_RESULT_DIR=/tmp/cpp-llm-cuda-profile \
bash code/chapter09_performance/scripts/run_cuda_profile.sh
```

结果目录包含：

| 文件 | 阅读目的 |
| --- | --- |
| `metadata.txt` | 核对模型、负载、GPU、驱动、CUDA 和正确性 |
| `program-output.txt` | 核对真实生成结果 |
| `kernel*.csv` | 按总时间和实例数寻找高占比计算核 |
| `memory*.csv` | 查看 Host/Device 搬运与显存操作时间 |

先按计算核总时间排序，再同时观察实例数。总时间高且实例数少，通常适合优化单次执行；总时间高且实例数很多，还应检查能否通过组批、持久化工作区或算子融合减少启动次数。报告中的单次平均时间不能代替端到端墙钟，因为服务还包含 Host 编排、同步和数据搬运。

需要保留可视化时间线时，可以直接运行 `nsys profile`。下面使用现成可执行文件表示命令结构，`<executable>` 和参数应替换为已通过正确性检查的固定负载：

```bash
"${CUDA_HOME}/bin/nsys" profile \
  --trace=cuda,nvtx,osrt \
  --sample=none \
  --cpuctxsw=none \
  --output=/tmp/cpp-llm-profile \
  <executable> <fixed-workload-arguments>

"${CUDA_HOME}/bin/nsys" stats \
  --report cuda_gpu_kern_sum \
  --format csv \
  /tmp/cpp-llm-profile.nsys-rep
```

时间线重点检查三个关系：Host 是否频繁同步等待 GPU，小计算核是否被大量重复启动，内存复制是否与计算串行。它们分别对应异步边界、组批或融合机会、数据驻留策略。

## 用 ncu 解释热点计算核

确认量化 GEMV 是热点后，运行仓库中的固定 tile 与自适应 tile 对照。脚本先执行正确性测试和配对计时，再为同一矩阵、格式、批大小和路径采集一个计算核：

```bash
CUDA_HOME=/usr/local/cuda-12.8 \
CHAPTER09_CUDA_ADAPTIVE_GEMV_PROFILE=1 \
CHAPTER09_CUDA_ADAPTIVE_GEMV_RESULT_DIR=/tmp/cpp-llm-adaptive-gemv-ncu \
bash code/chapter09_performance/scripts/run_cuda_adaptive_gemv_ab.sh
```

`ncu-summary.json` 汇总两条路径，原始 `ncu-*.csv` 保留 Compute、Instruction、Memory、Scheduler 和 Warp State 五组信息。阅读时按下面的因果顺序进行：

| 现象 | 优先核对 | 可能原因 |
| --- | --- | --- |
| DRAM 或 L1/TEX 吞吐高，计算管线利用率低 | Memory Workload、缓存命中率 | 权重读取、布局或访存合并受限 |
| 指令数量高，吞吐没有同步增加 | Instruction Stats | 重复解码、无效 tile 行或地址计算过多 |
| 活跃 Warp 少 | Occupancy、寄存器和共享内存 | 每个线程块资源占用过高 |
| Eligible Warp 少且 stall 高 | Scheduler、Warp State | 数据依赖、内存等待、同步或屏障过多 |

一次只比较一个受控变量，并确认计算核名称、矩阵形状、量化格式、批大小和启动次数一致。`ncu` 可能多次重放计算核，采集耗时和程序墙钟会显著放大，所以这些 CSV 用于原因归因，不进入端到端延迟结论。

## 从剖析回到优化决策

以 SmolVLM 双请求 Packed Prefill 为例，第一版把 Linear 和 FFN 的词元行合并，却逐词元启动 Attention。`nsys` 显示 Score、Softmax 和 Value 各启动 `12096` 次。改为每个请求一次多查询 Attention 后，各自降到 `960` 次，但端到端仍比串行路径慢约 `7.7%`～`11.7%`；剩余时间主要落在 Q8_0 GEMV 和行重排。因此 Packed 路径保留为显式实验路径，默认自适应路径继续选择串行 Context Prefill。

这个案例说明，组批不等于融合，也不保证加速。性能决策需要依次回答：

1. `nsys` 是否确认启动次数或热点分布发生预期变化。
2. `ncu` 是否解释了热点计算核的资源限制和变化原因。
3. 相同负载的端到端配对 A/B 是否改善。
4. 输出、缓存隔离和失败回滚是否仍通过正确性门禁。

本教程先运行已固定负载的脚本，避免手工命令遗漏模型和正确性闸门：

```bash
CHAPTER09_CUDA_PAGED_ATTENTION_PROFILE=1 \
bash code/chapter09_performance/scripts/run_cuda_paged_attention_ab.sh
```

脚本使用 `nsys profile` 生成临时报告，再用 `nsys stats --report cuda_gpu_kern_sum` 导出 Kernel 总时间、实例数和分布。结果先判断热点是否值得继续；EXP-GPU-07 中量化 GEMV 占约 `99.2%`，采用批次权重复用和 Warp 归约后的 EXP-GPU-08 仍占约 `96.5%`。因此分页 Attention 即使局部加速，也不是当前端到端首要瓶颈。

定位到 GEMV 后，再用 NCU 比较同一 kernel 的 cycles、共享内存、寄存器和 L1/TEX 指标。需要解释原因时继续采集 Compute、Instruction、Memory、Scheduler 和 Warp State section，核对指令数量、活跃与可调度 Warp、主要停顿和缓存事务。EXP-GPU-14 中固定 4 行与自适应 tile 采用相同矩阵、相同 launch 次数和独立进程；自适应实现的指令数减少约 `21%`～`37%`，共享内存减少 `50%`～`75%`，但 L1 命中率基本不变。这组指标把原因定位为减少无效 tile 行工作，而不是缓存策略改变。EXP-GPU-15 随后用真实 Decode 得出 `KEEP`，EXP-GPU-16 再确认收益覆盖 batch 1/2、上下文 16/128 和 Decode 1/8 步；三层证据共同支持默认路径变更。

确定单个 Kernel 是热点后，再对最小可执行负载使用 `ncu`。下面的命令只展示调用结构；Kernel 名称和可用指标会随 CUDA、GPU 架构与 Nsight Compute 版本变化，运行前先用 `ncu --query-metrics` 核对本机支持项。

```bash
ncu --set basic \
  --kernel-name regex:quantized_gemv_kernel \
  --launch-count 1 \
  <benchmark-executable> <model.gguf>
```

若出现 `ERR_NVGPUCTRPERM`，表示当前用户没有读取 NVIDIA GPU 性能计数器的权限。此时仍可使用 `nsys` 判断 kernel 时间占比，但 L2/DRAM 吞吐、占用率和 Warp stall 需要管理员开放 profiling counters 后再采集；不要用缺失的计数器推断具体硬件原因。

EXP-GPU-09～12 展示了 NCU 指标怎样改变优化方向。元数据广播和 shared payload 看似减少全局读取，却增加了 shuffle 或 barrier，最终 cycles 回退；原生 half 转换直接删除手写规格化分支，使 Q4_K kernel cycles 下降约 `28.8%`。判断时同时读取 `Elapsed Cycles` 与 `SM Frequency`，避免把动态时钟变化误判为代码收益。

分析完成后必须回到相同负载的端到端 A/B：`nsys` 决定热点优先级，`ncu` 解释硬件原因，墙钟样本和正确性闸门决定实现是否保留。

## 记录 GPU 运行状态

消费级 GPU 会随电源模式、温度和其他进程改变时钟。实验至少记录 GPU 名称、驱动、CUDA Toolkit、图形/显存时钟、功耗、温度和利用率；采集前后状态差异明显时，不把两轮墙钟结果直接归因于代码变化。`nvidia-smi` 快照只能描述采样时刻，严格实验还需要固定电源环境并持续采样。

## 验证阶段计时入口

有 TinyLlama GGUF 时，运行：

```bash
bash code/chapter09_performance/scripts/run_real_evidence.sh profile
```

脚本调用 <a href="https://github.com/datawhalechina/cpp-llm-engine/blob/main/code/engine/scripts/run_real_decoder_profile.sh"><code>run_real_decoder_profile.sh</code></a>，默认重复三轮，并在结果目录的 `profile/tinyllama-decoder-profile.txt` 保存 Decoder 阶段证据。文件先记录 experiment、负载、模型、CPU、编译器和构建类型，再记录正确性与阶段平均耗时；这些身份字段缺失时脚本直接失败。当前三轮归档结果见 <a href="https://github.com/datawhalechina/cpp-llm-engine/blob/main/code/chapter09_performance/results/profile/tinyllama-decoder-scalar.txt"><code>tinyllama-decoder-scalar.txt</code></a>。
