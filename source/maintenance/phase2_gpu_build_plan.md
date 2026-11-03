# 第二期 GPU 建设与学习计划设计

本文供教程维护使用。第二期学习者计划发布前，至少要具备可运行的 CUDA 环境检查、DeviceBuffer/stream 练习、CUDA kernel 测试和 cuBLAS MatMul backend；不能用接口草图代替已验证代码。

## 阶段目标

第二期不复制第一期的 Tokenizer、模型加载、Request 和 ServiceCore。它复用 CPU 已稳定的 contract、输入和 golden result，只替换 device memory、Backend、Kernel 和异步执行路径。

完成第二期后，学习者应能回答四个问题：

1. host/device 数据由谁拥有，何时复制和释放？
2. 同一 Operator 如何派发 CPU reference 与 CUDA backend？
3. GPU 结果怎样与 CPU golden 对照，时间怎样用 event 正确测量？
4. 单算子收益如何传导到 Decoder、KV Cache 和服务吞吐？

## 硬件定位与发布口径

第二期面向单机、单张 NVIDIA 消费级显卡，TinyLlama 1.1B Q4_K_M 是首个端到端模型。CPU backend 必须继续作为正确性基线；多 GPU、模型并行、NCCL 和分布式控制面不进入第二期。

`8 GB` 只作为首期实现的设计验证目标，不写成最低要求；`12 GB` 只作为扩展配置。CUDA 路径完成后，分别在 4/6/8 GB 显存限额或对应设备上测试，再更新学习者环境建议。

| 必须记录 | 用途 |
| --- | --- |
| GPU、驱动、Toolkit、compute capability | 固定运行环境 |
| 模型、量化格式、context、并发数 | 固定 workload |
| 权重、KV Cache、workspace、CUDA runtime | 解释显存组成 |
| peak allocated、OOM 边界、释放后占用 | 确定最低要求和资源回收 |

## 建设优先级

GPU 建设按依赖关系推进。每一级必须同时提供正文、代码、测试、benchmark 和学习记录字段，正确性失败时不进入性能比较。

| 优先级 | 建设项 | 可验证产物 | 进入下一项的门槛 |
| --- | --- | --- | --- |
| P0-1 | 环境与外部 GPU baseline | driver/toolkit 检查；同一 TinyLlama 的 llama.cpp CUDA 输出 | `nvidia-smi`、最小 kernel 和真实 GGUF GPU 推理均可运行 |
| P0-2 | DeviceBuffer、拷贝与 stream/event | RAII device memory 练习；同步/异步拷贝测试；event 计时 | 无泄漏、越界和隐式生命周期；CPU/CUDA 结果一致 |
| P0-3 | 首个自研 CUDA kernel backend | Vector Add 或简单 elementwise 的 CPU/CUDA 共用 contract | 非整除 grid、空输入、错误设备和数值误差测试通过 |
| P0-4 | cuBLAS MatMul backend | `MatmulBackend` 的 CUDA 实现、row/column-major 映射和 CPU 对照 | 多组 shape、transpose、非法输入、event benchmark 通过 |
| P1-1 | GPU Tensor/Workspace 执行边界 | device、stream、workspace ownership 与复用策略 | 上层 Operator 不直接调用 `cudaMalloc/cudaFree` |
| P1-2 | Quantized Linear GPU 基线 | Q4_K/Q6_K dequantize + GEMM 基线；再评估 fused kernel | 与 CPU/llama.cpp 误差一致，先有非融合基线再做融合 |
| P1-3A | RoPE kernel | 连续 `[tokens, heads, head_dim]`、position offset、局部 rotary dimension | 多 shape CPU golden、错误参数和 event 计时通过 |
| P1-3B | 连续 device KV payload | 每层 K/V 容量、append/read/reset 和请求级所有权 | 容量边界、数据保持、请求隔离和释放通过 |
| P1-3C | causal GQA Attention | prefill/decode、causal mask、GQA head 映射和连续 device K/V | CPU golden、历史 KV 和非整除 grid 通过 |
| P1-4A | Decoder 缺失算子 | RMSNorm、Residual、SwiGLU | 257/2048 宽度 CPU golden 与错误参数通过 |
| P1-4B | 合成单层 GPU Decoder | 按真实顺序组装已有 backend，并保留中间 Tensor | 单 token CPU/GPU 逐阶段一致 |
| P1-4C1 | 统一 Linear 权重视图 | Float32 与 Q4_K/Q6_K 使用同一 Decoder 调度入口 | 量化输出可直接写 rank-3 Q/K/V Tensor |
| P1-4C2 | GGUF layer-0 上传 | F32 norm 与九个首层权重按格式上传，不永久反量化 | shape、格式、payload bytes 和显存所有权通过 |
| P1-4C3 | 真实首层到完整 Decoder | TinyLlama 首层、N 层、22 层依次接入 | top-k/golden token、显存释放和重复运行通过 |
| P2-1 | Runtime 异步化 | stream、event、copy/compute overlap 和 batch 执行 | 不改变 Request/Scheduler 语义，指标能区分同步等待 |
| P2-2 | 服务优化 | GPU benchmark、memory pressure、CUDA Graph 可选实验 | correctness、TTFT、吞吐、显存峰值和失败回收齐全 |

## 第二期学习任务结构

学习顺序压缩为六个必做任务和一个选做任务。学习重点从“CUDA API 会不会写”转向“设备所有权、异步依赖、backend 替换和证据闭环”。

| 顺序 | 学习任务 | 侧重点 | 验收产物 |
| --- | --- | --- | --- |
| 1 | 固定 GPU 环境与基线 | driver、toolkit、compute capability、外部 GPU 推理 | 环境记录与同 prompt CPU/GPU baseline |
| 2 | 掌握 device memory 与异步执行 | RAII、H2D/D2H、pinned memory、stream、event | DeviceBuffer 测试和正确 event 计时 |
| 3 | 实现首个 CUDA backend | grid/block、边界、错误传播、CPU golden | 共用 contract 的 elementwise backend |
| 4 | 实现 cuBLAS MatMul | layout、transpose、dtype、handle/workspace | CPU/cuBLAS 数值对照与多 shape benchmark |
| 5 | 接入模型关键 kernel | Quantized Linear、RoPE、GQA Attention、device KV | 单层 prefill/decode 的 CPU/GPU 对照 |
| 6 | 跑通 GPU Decoder | 共享 Model、请求 Context、22 层 forward、显存生命周期 | 固定 TinyLlama golden 与显存记录 |
| 7 | Runtime 与服务优化，可选 | overlap、continuous batching、CUDA Graph、服务指标 | 单变量 A/B，包含 TTFT、吞吐和显存峰值 |

## 内容落点

GPU 内容优先补入已有职责位置，避免先新增一批平行章节。只有当 GPU Decoder 和 Runtime 内容使现有章节无法保持清晰时，再拆出独立 GPU 章节组。

| 内容 | 首选位置 | 理由 |
| --- | --- | --- |
| 环境、DeviceBuffer、stream/event 小题 | Foundation 与 `exercises/foundation` | 先隔离 CUDA 工程和资源管理能力 |
| CUDA/cuBLAS backend | Chapter03 | 复用 `MatmulBackend` 和 CPU reference |
| Quantized Linear CUDA | Chapter05 | 复用 Q4_K/Q6_K 格式与 correctness 基线 |
| Attention、RoPE、device KV | Chapter04 | 保持模型数学和执行 backend 的对照 |
| GPU Model/Context adapter | Engine | 不让 Runtime 依赖具体模型与设备实现 |
| GPU benchmark/profiling | Chapter07-08 | 复用 workload、指标和单变量实验方法 |
| GPU 服务执行策略 | Chapter09 | 只扩展 executor 和资源指标，不复制 ServiceCore |

## 第一批实现范围

第一批只完成 `DeviceBuffer -> stream/event -> CUDA elementwise backend -> cuBLAS MatMul backend`。暂不同时实现 FlashAttention、量化 fused kernel、CUDA Graph 或多 GPU；这些能力依赖稳定的 device ownership 和测量方法。

当前 P1-1、P1-2、P1-3A/B/C、P1-4A/B 和 P1-4C1 已完成。`CudaLinearWeight` 统一 Float32 与 Q4_K/Q6_K 分派，量化输出已验证可直接写 rank-3 Tensor，反量化 workspace 不成为永久权重副本。下一步 P1-4C2 从 GGUF 上传 layer-0 的 norm 和 packed 权重，再由 P1-4C3 执行真实首层、N 层和 22 层。

宿主机已完成 CUDA 运行门槛验证：RTX 5070 Laptop、12 GB 显存、R570.211.01 驱动与 CUDA Toolkit 12.8。一次性探针已通过设备枚举、`cudaMalloc`、H2D、单元素 kernel、同步和 D2H；CUDA 13.0 因要求 R580 驱动而不作为当前项目工具链。受控沙箱内仍可能看不到 `/dev/nvidia*`，GPU 测试需要在允许访问设备节点的宿主环境运行。
