# 08.4 分析 Kernel 启动与 CUDA Graph

Decode 会反复提交形状较小的 Kernel。只有时间线证明 Host 提交、同步和 Kernel 间空隙占比显著，并且计算图足够稳定时，CUDA Graph 才是合理实验对象。

## 先判断瓶颈类型

大 Kernel 占满时间时应先优化计算实现；大量短 Kernel 和稳定重复的提交序列才适合比较 Graph replay 或 Kernel fusion。

| 观察结果 | 优先动作 |
| --- | --- |
| GEMM/Attention Kernel 占主导 | 优化 Kernel、布局、数据类型或 backend |
| 大量短 Kernel 与 launch gap | 建立 CUDA Graph 单变量 A/B |
| Host/Device 同步密集 | 减少同步和往返复制 |
| shape、地址或控制流频繁变化 | 先稳定 Runtime 与 Buffer 生命周期 |
| 固定地址和重复 Decode 图 | 评估 capture cache 与 replay |

## 区分教学 Trace 与 GPU Trace

Chapter07 的阶段、事件和 Chrome Trace 用于学习分析顺序，不包含真实 CUDA launch。GPU 结论需要 Nsight Systems 的 Host/Stream 时间线和 Nsight Compute 的 Kernel 指标。

运行 `bash code/cuda/scripts/run.sh` 可以验证当前 CUDA correctness，但不能单独证明 launch overhead。真实 Graph 实验还要固定模型、token 数、batch、stream、buffer 地址和 correctness golden，并记录 capture 成本与 replay 次数。

| 证据 | 教学 Trace 能回答 | 真实 GPU 还需回答 |
| --- | --- | --- |
| Phase summary | prefill/decode 哪段较重 | CPU/GPU 是否重叠 |
| Event timeline | 事件先后和同步位置 | launch、memcpy、stream gap |
| Hotspot | 先调查哪个事件 | Kernel self time、调用次数和占用率 |
| Graph A/B | 不提供性能结论 | capture 成本能否被 replay 摊销 |

## 验证与衔接

保留 Graph 路径需要 correctness 通过、稳态 decode 达到预设阈值、动态 shape 能选对图，并在无法 replay 时安全回退。实验设计按 [`08.5 A/B 对比实验`](./05_ab_experiment.md) 固定唯一变量，不能同时修改 Graph 与 fusion 后再声称某一项有效。
