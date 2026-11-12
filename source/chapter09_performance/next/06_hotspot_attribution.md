# 09.6 从阶段耗时定位到热点

热点归因沿着“总时间 → 阶段 → 函数或 Kernel → 硬件原因”逐步缩小。每一步都要保留上一级测量，避免局部加速后端到端结果没有变化。

| 层级 | 可观察对象 | 下一步判断 |
| --- | --- | --- |
| 请求 | TTFT、ITL、完成时间 | Prefill、Decode 还是等待 |
| Runtime | 模型计算、采样、调度、输出 | 计算路径还是系统开销 |
| Decoder | QKV、Attention、FFN、LM Head | 哪组算子占主导 |
| 算子 | GEMM、归一化、RoPE、Softmax | 计算、带宽或布局问题 |
| Kernel/函数 | 指令、访存、同步、launch | 是否值得改实现 |

下图展示一次完整的下钻顺序。只有上一级证据确认范围后，才进入更细的函数、Kernel 或硬件指标；局部优化完成后还要回到请求级指标复测。

![从端到端时间下钻到硬件原因](../../assets/diagrams/chapter09-hotspot-attribution.svg)

## 用数据解释而不是猜测

热点表至少记录符号、调用次数、自耗时、总耗时和对应 workload。没有工作负载身份的热点截图不能作为本教程的优化结论。

| 证据 | 可以支持的判断 | 不能直接推出 |
| --- | --- | --- |
| 阶段占比高 | 优先继续分析该阶段 | 该阶段一定能优化 |
| 内存带宽接近上限 | 访存可能主导 | 增加线程一定更快 |
| Kernel launch 数量多 | 启动开销值得核对 | 融合后一定改善端到端延迟 |
| SIMD/CUDA 路径更快 | 当前 workload 下局部有效 | 所有模型和 shape 都更快 |

## 记录归因证据

先运行 `bash code/chapter09_performance/scripts/run_real_evidence.sh profile`，确认阶段文件含有 `WL-CPU-TINYLLAMA-DECODER-PHASE-V1`；需要继续下钻 CPU 函数时，再对同一可执行文件和输入使用 `perf record`。归因结果应在[实验资产索引](../../reference/experiment_assets.md#exp-cpu-04)中指向原始文件和复现入口，而不是只保留截图。

当阶段、函数和硬件指标支持同一个优化假设时，进入 `09.7`；若不能回指相同 workload，不得进入 A/B 结论。

DeepSeek CUDA 的 grouped MoE 实验展示了为什么必须逐层下钻。EXP-GPU-17 的首轮结果中，专家路由重叠率为 `100%`，但端到端收益方向不稳定。使用 CUDA Profiler capture range 只捕获一次 grouped Decode 后，Nsight Systems 记录到 54 次 MLA kernel，总时间约 `32.16 s`，占 GPU kernel 时间 `99.9%`。`cudaStreamSynchronize` 的 Host 等待也接近同一总量，表示 Host 在等待 MLA 完成，而不是同步调用自身消耗了这些计算时间。

依据该证据，CUDA MLA 改为 score、Softmax、Value 聚合三阶段，并让概率在 Value 维之间复用。优化后 6 轮反向顺序配对中，顺序与 grouped Decode 中位数分别为 `550.2 ms` 和 `455.0 ms`，logits 与词元门禁通过。新的单次捕获中，score kernel 合计约 `317.6 ms`、占 kernel 时间 `82.3%`，Value 聚合约 `5.44 ms`。这次复测同时回答了两个问题：三阶段 MLA 应保留；下一轮应继续拆解 score 的 latent 投影，而不是优先融合 MoE。
