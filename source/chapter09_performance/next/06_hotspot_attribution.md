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

长度矩阵还能提示热点何时迁移。[EXP-GPU-19](../../reference/experiment_assets.md#exp-gpu-19) 固定模型、连续 MLA 和 27 层计时范围，只改变 Prefill 长度。Tensor/逐词元比例从长度 2 的 `0.552` 降到长度 32 的 `0.183`，长度 128 时回升到 `0.343`。随后只捕获一次 Tensor Prefill：长度 32 的 MLA score/value 占 Kernel 时间约 `62.2%`，量化 GEMV 约占 `37.6%`；长度 128 时两者变为约 `88.5%` 和 `11.4%`。这组原因指标确认长上下文下因果 Attention 正在抵消一部分权重复用收益，下一轮应优化 score/value 的数据复用，而不是继续优先优化 MoE GEMV。

[EXP-GPU-20](../../reference/experiment_assets.md#exp-gpu-20) 随后只改变 MLA 计算顺序。权重吸收先计算 `q_nope @ W_K`，Value 阶段先聚合概率加权 latent，再执行 `W_V @ c_attended`。长度 32 和 128 的 absorbed/legacy 中位数比例为 `0.515/0.169`，logits 最大误差不超过 `5.73e-5`，词元一致。长序列收益更明显，与被移除的重复计算随历史长度增长相符，因此 absorbed 路径进入默认 Factory，legacy 路径保留为回归基线。

替换默认路径后还要重新定位热点。吸收式路径在长度 32 和 128 下，量化 GEMV 分别约占 Kernel 时间的 `94.2%` 和 `91.6%`，MLA score/value 降至约 `5.1%` 和 `8.0%`。长度 128 的 `38,048` 次复制中，Device-to-Device 为 `30,290` 次、约占复制 GPU 时间的 `65.9%`；Device-to-Host 和 Host-to-Device 分别为 `4,376` 和 `3,382` 次。调用点与数量共同指向逐行 `pack/unpack`，下一轮应先用设备端索引收集和加权散回减少小复制，再用端到端 A/B 决定是否保留。

[EXP-GPU-21](../../reference/experiment_assets.md#exp-gpu-21) 保持 Router、Top-K、专家 Linear 和 absorbed MLA 不变，只把 MoE 分发从逐行复制改为设备端 gather/scatter。长度 32/128 的 packed/row-copies 中位数比例为 `0.955/0.947`，最大 logits 误差不超过 `3.25e-5`，词元一致。长度 128 的 Device-to-Device 与 Device-to-Host 复制分别降到 `338` 和 `27` 次，两个新 Kernel 各约占 `0.1%`；量化 GEMV 仍约占 Kernel 时间的 `92%`。因此 packed-dispatch 进入默认 Factory，后续优化回到量化 GEMV，而不是继续拆分行调度。

回到 GEMV 后，先按量化格式拆开原因。[EXP-GPU-22](../../reference/experiment_assets.md#exp-gpu-22) 显示 Q2_K batch=1 约 `63.4%` 的 Warp 周期在等待 L1TEX scoreboard；IQ4_NL 则约 `80%` 消耗在 barrier。前者对应 256 元素 super-block 的读取与解码，后者来自用 256 线程处理 32 元素 block，优化入口并不相同。

[EXP-GPU-23](../../reference/experiment_assets.md#exp-gpu-23) 只把 IQ4_NL 改为每个输出行一个 32 线程 Warp。batch=1 的执行指令从约 `6.62M` 降到 `2.27M`，显存吞吐从 `10.59 GB/s` 升至 `29.51 GB/s`；batch=1/2/4/8 的微基准均改善且 CPU reference 最大误差为 `0`。这组结果解释了 kernel 为什么变快，但还不能决定真实模型默认值。

[EXP-GPU-24](../../reference/experiment_assets.md#exp-gpu-24) 因此固定 absorbed MLA、packed MoE、模型和输入，只切换 IQ4_NL GEMV。长度 32/128 的单 Warp/通用路径中位数比例为 `0.749/0.848`，logits 最大误差为 `0`，支持把单 Warp 路径下发到请求 Context 和 batch workspace。下一轮应继续分析 Q2_K 的 payload 访问，而不是继续缩小 IQ4_NL 线程块。

[EXP-GPU-25](../../reference/experiment_assets.md#exp-gpu-25) 保留 IQ4_NL 路径，只重排 Q2_K 的工作分配。一个 Warp 读取 64-byte quant payload，每个 lane 从 packed byte 提取四个 2-bit 值，scale 由 lane 0/16 广播。batch=1 的执行指令约从 `5.75M` 降至 `2.81M`，Warp cycles 从 `25.40` 降至 `8.96`，显存吞吐从 `19.78` 升至 `40.82 GB/s`；batch=1/2/4/8 两轮微基准均改善，CPU reference 最大误差为 `0`。

[EXP-GPU-26](../../reference/experiment_assets.md#exp-gpu-26) 进一步把 IQ4_NL 单 Warp 设为双方共同基线，只比较 Q2_K。长度 32/128 的新旧路径中位数比例为 `0.750/0.795`，最大 logits 误差不超过 `4.01e-5`，词元一致。因此 block-aware 路径进入 DeepSeek Factory 默认值；`AdaptiveIq4Warp` 留作 Q2_K 单变量基线，通用 `Adaptive` 留作两种格式的初始基线。

默认值切换后要再次剖析，而不是沿用优化前的热点。[EXP-GPU-27](../../reference/experiment_assets.md#exp-gpu-27) 固定 absorbed MLA、packed MoE 和 block-aware GEMV，重新捕获长度 32/128 的完整 Prefill。Q2_K 单 Warp 分别占 Kernel 时间约 `44.4%/44.8%`，IQ4_NL 单 Warp 约占 `21.9%/21.1%`；仍走通用路径的 Q3_K 上升到 `13.8%/18.1%`。MLA score/value 合计约为 `9.4%/13.0%`。因此下一轮先分析 Q3_K 的位平面解码与工作分配，MLA 暂不继续拆分。

[EXP-GPU-28](../../reference/experiment_assets.md#exp-gpu-28)让一个 Warp 复用 Q3_K 的 low-2-bit byte、high-bit mask 和 6-bit group scale。batch 1 的执行指令约从 `10.29M` 降至 `6.52M`，Warp cycles 从 `21.94` 降至 `15.02`，显存吞吐从 `14.06` 升至 `15.81 GB/s`；batch 1、4、8 两轮时间均改善，batch 2 对执行顺序敏感。因此微基准支持进入真实模型复测，但不单独决定默认值。

[EXP-GPU-29](../../reference/experiment_assets.md#exp-gpu-29)固定 Q2_K 与 IQ4_NL 单 Warp，只切换 Q3_K。长度 32/128 的新旧路径中位数比例为 `0.911/0.935`，最大 logits 误差不超过 `3.72e-5`，词元一致。Q3_K 专用路径因此进入 Factory 默认值；此前的 block-aware 路径保留为单变量回归基线。

[EXP-GPU-30](../../reference/experiment_assets.md#exp-gpu-30)随后对 all-Warp 默认路径再次剖析。长度 32/128 中，Q2_K 与 IQ4_NL 合计约占 Kernel 时间 `65.1%/69.1%`，Q3_K 约占 `15.5%/13.4%`，MLA score/value 合计约为 `7.6%/13.9%`。同时，`cudaMemcpyAsync` 占 CUDA API 时间约 `91.0%/92.5%`；长度 128 出现 `27` 次 Device-to-Host 大块复制和 `3117` 次小型 Host-to-Device 复制。代码反查表明，每层 packed MoE 都把 `[T,D]` hidden 下载到 CPU 计算 Router/Top-K，再上传专家行索引与权重。因此下一轮优先迁移设备端 Router 与分发表，而不是继续微调已验证的 GEMV。

[EXP-GPU-31](../../reference/experiment_assets.md#exp-gpu-31)先迁移 Router 点积、全专家 Softmax、稳定 Top-K 和权重归一化。候选路径每层只回传 `[T,K]` 专家编号与权重，Host 继续建立专家分组，因此本轮单变量是路由计算位置。长度 32/128 的 device-router/host-router 中位数比例为 `0.957/0.941`，最大 logits 误差不超过 `3.91e-5`，词元一致。设备 Router 因此进入默认 Factory；下一轮 profile 应核对 27 次大块 D2H 是否消失，并判断剩余小型 H2D 是否支持继续迁移专家分组表。

[EXP-GPU-32](../../reference/experiment_assets.md#exp-gpu-32)重新捕获设备 Router 默认路径。长度 128 的 D2H 从 `27` 次、`27.673 MB` 变为 `53` 次、`0.569 MB`：次数增加来自每个 MoE 层分别读取专家编号和权重，但完整 `[T,D]` hidden 回传已经消失，传输字节减少约 `97.9%`。`3117` 次小型 H2D 保持不变，说明 Host 建立的专家行号和权重仍被逐组上传。量化 GEMV 约占 Kernel 时间 `83.6%`，设备 Router 约占 `1.0%`。下一项数据面优化应让设备端生成专家分组元数据；Router 算术暂不继续融合。

[EXP-GPU-33](../../reference/experiment_assets.md#exp-gpu-33)让设备端按专家扫描 `T×K` 路由表，并直接生成 gather/scatter 使用的行号和权重。候选与设备 Router 默认路径的 logits 和词元完全一致，但长度 32 的中位数比例为 `1.225`，长度 128 为 `0.998`。短序列回退说明本轮增加的全路由扫描、kernel launch 和专家计数回读抵消了小型 H2D 的节省。该路径保留用于教学和后续融合实验，默认值不变；下一次尝试需要把分组、专家执行或工作区选择一起重构，而不是继续叠加独立扫描 kernel。
