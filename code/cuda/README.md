# CUDA 公共基础层

`code/cuda` 提供独立 CUDA backend，不让 CPU 默认回归依赖 CUDA Toolkit。当前已完成资源层、Linear、RoPE、连续与分页 K/V、causal GQA，以及 Decoder 组装所需的 RMSNorm、Residual 和 SwiGLU 算子。

## 资产总览

| 资产 | 作用 |
| --- | --- |
| `cuda_status.h/.cpp` | 保存 CUDA 错误码、操作名和可读错误，并转换为 core `Status` |
| `device_buffer.h/.cpp` | move-only 设备字节缓冲区、同步/异步 H2D/D2H 和显式释放 |
| `pinned_host_buffer.h/.cpp` | 为可重叠异步复制提供页锁定 host 内存 |
| `cuda_stream.h/.cpp` | move-only 非阻塞 stream、同步和显式销毁 |
| `cuda_event.h/.cpp` | move-only event、记录、同步和设备 elapsed time |
| `cuda_runtime_test.cu` | 真实设备资源、同步/异步流水线、计时和失败路径测试 |
| `elementwise_backend.h/.cu` | 使用公共 VectorAdd contract 的 CUDA kernel backend |
| `cuda_elementwise_test.cu` | CPU golden、grid 边界、错误参数和多轮 event timing |
| `cublas_status.h/.cpp` | 保留原始 cuBLAS 状态码、操作名和 core Status 转换 |
| `cublas_matmul_backend.h/.cpp` | move-only handle 和 row-major SGEMM 适配 |
| `cuda_cublas_matmul_test.cu` | 多 shape CPU golden、非法输入、移动和 event timing |
| `cuda_tensor.h/.cpp` | 连续 Float32 CUDA Tensor 与公共 Tensor contract |
| `cuda_workspace.h/.cpp` | 只在容量不足时增长的可复用临时显存 |
| `cuda_execution_context.h/.cpp` | 统一持有 stream、cuBLAS handle、workspace 和 backend 调度 |
| `cuda_execution_context_test.cu` | Tensor 复制、批量行打包/拆包、算子调度、workspace 复用和移动测试 |
| `cuda_quantized_matrix.h/.cpp` | packed Q4_K/Q6_K row-major device 权重 |
| `quantized_dequantize_backend.h/.cu` | Q4_K/Q6_K 到 Float32 workspace 的 CUDA 解码 |
| `quantized_gemv_backend.h/.cu` | 单 token Q4_K/Q6_K 解码、点积与行归约融合 |
| `cuda_quantized_linear_test.cu` | CPU reference、融合 GEMV、批量回退、错误路径和固定 workload 计时 |
| `quantized_gemv_matrix.cu` | Q4_K/Q6_K 的 batch=1/2/4/8 专项矩阵与逐行正确性对照 |
| `core/rope_contract.h` | 固定原地 RoPE 的连续布局、旋转维度、位置和频率参数 |
| `rope_backend.h/.cu` | 每线程处理一个相邻分量对的 CUDA RoPE backend |
| `cuda_rope_test.cu` | 多 token/head CPU golden、局部旋转、错误参数和 event 计时 |
| `cuda_kv_cache.h/.cpp` | 请求级连续 K/V device storage、D2D append、clear 和显式释放 |
| `cuda_kv_cache_test.cu` | append 偏移、容量、shape、移动、复用和请求隔离 |
| `cuda_paged_kv_cache.h/.cpp` | 按 layer/block/slot 保存共享设备分页 K/V payload |
| `cuda_paged_kv_cache_test.cu` | Chapter07 非连续 BlockTable、分页 GQA、清块和 CPU golden |
| `core/gqa_attention_contract.h` | 固定 Q/KV head 映射、causal position、scale 和布局 |
| `gqa_attention_backend.h/.cu` | score、稳定 softmax、value aggregation 三阶段 CUDA backend |
| `cuda_gqa_attention_test.cu` | prefill/decode CPU golden、错误路径和 event 计时 |
| `core/decoder_elementwise_contract.h` | RMSNorm 与 SwiGLU 连续向量契约 |
| `decoder_elementwise_backend.h/.cu` | 单 block RMSNorm 和逐元素 SwiGLU kernel |
| `cuda_decoder_elementwise_test.cu` | 257/2048 宽度 CPU golden 与错误 shape |
| `cuda_decoder_layer.h/.cpp` | 单 token Llama Decoder 的算子编排与中间 Tensor 所有权 |
| `cuda_decoder_batch_layer.h/.cpp` | 合并批量无状态算子并保留逐请求 Attention/KV 的 Decoder 编排 |
| `cuda_decoder_layer_test.cu` | 合成小维度逐阶段 CPU golden 与 KV 增长 |
| `cuda_linear_weight.h` | Float32 Tensor 与 Q4_K/Q6_K Matrix 的统一只读 Linear 视图 |
| `scripts/run.sh` | 显式选择 CUDA Toolkit，隔离构建并运行 CTest |

## 所有权与失败路径

`DeviceBuffer` 独占一次 `cudaMalloc` 返回的地址，不允许复制，只允许移动。零长度缓冲区是合法空对象；非零复制必须同时满足 host 指针非空、offset 不溢出且范围不超过容量。析构函数只做兜底释放，需要观察释放错误时应显式调用 `reset()`。

| 操作 | 成功后的状态 | 失败时的保证 |
| --- | --- | --- |
| `allocate(bytes, output)` | `output` 独占新 allocation | 原 `output` 保持不变 |
| `copy_from_host` | 指定范围已同步写入 device | 越界和空指针在调用 CUDA 前拒绝 |
| `copy_to_host` | 指定范围已同步写回 host | 越界和空指针在调用 CUDA 前拒绝 |
| `copy_*_async` | 操作进入指定 stream，完成时间由 stream/event 确认 | 无效 stream、越界和空指针在入队前拒绝 |
| move | 新对象接管地址，源对象变空 | 不复制设备数据 |
| `reset()` | 地址和容量清零 | 返回 `cudaFree` 的原始错误 |

## 异步流水线与计时

测试使用 `PinnedHostBuffer`，按同一 stream 上的顺序执行 `Event(start) -> H2D -> kernel -> D2H -> Event(end)`。页锁定内存让 CUDA Runtime 可以直接进行异步 host/device 传输；普通 pageable 内存即使调用 `cudaMemcpyAsync`，也可能发生隐式 staging 或同步，不能作为重叠执行的稳定前提。

`CudaEvent::elapsed_time_ms()` 测得的是两个 event 在 GPU 时间线上的间隔，适合测 kernel 和复制流水线；它不等于用户看到的完整端到端延迟。等待结果时优先同步结束 event 或对应 stream，不在每个操作后调用 `cudaDeviceSynchronize()`。

## 首个 CUDA Backend

`CudaElementwiseBackend` 接收 core `VectorAddRequest` 和显式 `CudaStream`。请求只固定 `lhs + rhs -> output`、Float32 和元素数，指针地址空间由 backend 决定；CUDA 实现不让 core 或 CPU 路径包含 CUDA header。

测试用 `1/255/256/257/4099` 元素覆盖单线程、block 边界和非整除 grid，再以 CPU 逐元素结果作为 golden。固定 benchmark 使用 `1,048,576` 元素、1 次 warmup 和 10 次 event 计时，计时区间只包含 kernel，D2H 和结果检查位于区间之外。

`CublasMatmulBackend` 独占一个 move-only `cublasHandle_t`，调用时把 handle 绑定到显式 stream。core 的语义保持 row-major `C[M,N] = A[M,K] * B[K,N]`；实现不复制或转置数据，而是交换 A/B 参数，用 column-major 视角执行 `C^T[N,M] = B^T[N,K] * A^T[K,M]`。

## GPU 执行边界

`CudaTensor` 把公共 `TensorContract` 与 `DeviceBuffer` 绑定，首版只接受连续 Float32 CUDA Tensor。`CudaWorkspace` 把临时显存从算子中移出，`reserve()` 在容量足够时复用原地址，仅在增长时重新分配并递增计数。

`CudaExecutionContext` 独占一条 stream、一个 cuBLAS handle 和一个 workspace，并以 `CudaTensor` 分派 Vector Add 与 MatMul。上层调用者不再直接创建 `DeviceBuffer`、设置 cuBLAS stream 或在算子内部调用 `cudaMalloc/cudaFree`。

| 对象 | 拥有什么 | 不负责什么 |
| --- | --- | --- |
| `CudaTensor` | shape/stride/dtype/device contract、device storage | stream、kernel 和临时 workspace |
| `CudaWorkspace` | 可增长复用的临时 device storage | Tensor 语义和算子调度 |
| `CudaExecutionContext` | stream、cuBLAS handle、workspace、backend dispatch | Request、Scheduler 和 KV Cache 生命周期 |

## Quantized Linear

`CudaQuantizedMatrix` 保存 packed Q4_K/Q6_K 权重。单 token 的 `quantized_linear()` 由一个 CUDA block 负责一个输出行，在同一 kernel 内解码、乘加和归约，不创建完整 Float32 权重矩阵。Warp 内使用 shuffle，8 个 Warp 的部分和再由首个 Warp 合并。`quantized_linear_batch()` 接收 `[B,K]` 输入和 `[N,K]` 权重，把最多 4 个输入行组成一个 tile；同一线程解码一次权重后更新多行累加器，直接生成 `[B,N]`。该路径面向教学中的小批量 Decode，更大的 tile 需要继续核对寄存器压力和占用率。

NCU 基线显示 Q4_K kernel 的 L1TEX 等待明显，而 DRAM 吞吐和 occupancy 不是首要限制。EXP-GPU-09 的 Warp 元数据广播、EXP-GPU-10 的 shared payload 和 EXP-GPU-11 的显式 FMA 分别回退约 `10.1%`、`1.6%` 和 `4.3%` cycles，代码不保留这些候选。EXP-GPU-12 使用 `__ushort_as_half` 与 `__half2float` 替换手写 FP16 位解码，cycles 约减少 `28.8%`，真实双请求 logits 保持一致；当前代码保留原生 half 转换，后续优化重新聚焦 L1/TEX payload 访问。

固定 workload 使用 `rows=256`、`cols=256`、warmup 1 次和测量 10 次。计时在权重与输入上传后开始，包含融合 kernel、主机提交以及末尾同步，不包含 H2D/D2H。RTX 5070 Ti Laptop / CUDA 12.8 上，本次 Q4_K 与 Q6_K 主机均值约为 `0.0041 ms` 和 `0.0044 ms`；真实 TinyLlama 优化前后证据登记为 EXP-GPU-02/03。

EXP-GPU-06 记录初版批量融合路径：真实双请求分页 Decode 的两轮 Tensor Batch/顺序执行中位数比例为 `0.922/0.919`。EXP-GPU-08 将批次改为最多 4 行共享一次权重解码，并使用 Warp 归约；两轮比例为 `0.466/0.612`。EXP-GPU-12 再把手写 FP16 解码替换为 CUDA 原生 half 转换，顺序 Decode 降至 `35.12/30.54 ms`，Tensor Batch 降至 `19.80/17.02 ms`，logits 保持一致。这些结果支持当前小批量融合路径，不代表大 batch 量化 GEMM 已完成。

批次扩展使用独立矩阵入口：`bash code/chapter09_performance/scripts/run_cuda_quantized_gemv_matrix.sh`。可执行程序默认测量 `batch=1/2/4/8`，也可用 `--batch N --format q4|q6 --iterations N` 缩小 workload。EXP-GPU-13 中 batch 1→4 的 Q4_K/Q6_K 时间基本不变，batch=8 约翻倍，符合固定 `tile=4` 会启动两个 y 方向 block 的设计。参数只选择实验输入，不切换 kernel 实现；CUDA GEMV 当前只有通过门禁的默认路径，失败候选保留为实验资产而不进入运行时枚举。

## RoPE Kernel

RoPE 请求使用连续 `[tokens, heads, head_dim]` 布局，在原 Tensor 上旋转前 `rotary_dimension` 个分量。相邻两个分量组成一对，每个 CUDA 线程负责一对；连续序列使用 `position_offset + token_index`。Decode batch 使用另一入口显式上传 `positions[B]`，让不同长度请求在 `[B,heads,head_dim]` 中保持各自位置。`rotary_dimension` 可以小于 `head_dim`，未参与旋转的尾部保持不变。

测试先用同构 CPU 公式生成 golden，再覆盖多 token、多 head、非整除 grid、局部旋转维度、非连续位置 `{0,7,31}`、非法参数和 `CudaExecutionContext` shape 检查。RTX 5070 Ti Laptop GPU / CUDA 12.8 上，`257 x 32 x 64`、warmup 1 次、测量 10 次的本次 kernel event 均值为 `0.0083808 ms`；该数字只用于确认计时闭环，不是跨设备性能基线。

## 连续 Device KV Cache

`CudaKvCache` 分别独占 `[capacity_tokens, kv_heads, head_dim]` 的 K/V 连续显存。新 K/V 使用相同布局的 `CudaTensor` 输入，并在调用方 stream 上追加到当前 token 尾部；成功后才推进逻辑长度。`clear()` 只把长度归零以复用 allocation，`reset()` 才释放显存。

连续布局用于固定 prefill/decode 的数据语义和请求所有权，每个请求、每层持有独立缓存。分页 block、回收池和 prefix sharing 位于后文的分页路径，它们不改变这条数值基线。

## Causal GQA Attention

`CudaGqaAttentionBackend` 接收 Q `[Tq,Hq,Dh]` 和连续缓存 K/V `[Tk,Hkv,Dh]`。query head 按 `query_head / (Hq/Hkv)` 共享 KV head；`query_position_offset + query_token` 决定 causal mask 的最后可见位置。`CudaExecutionContext::append_kv()`、`gqa_attention()` 保证上传、D2D append 和计算进入同一 stream。

首版刻意保留三个 kernel：先生成 `[Tq,Hq,Tk]` score workspace，再逐行执行稳定 softmax，最后聚合 value。该路径用于核对 prefill/decode 和 GQA 数值，不是 FlashAttention。RTX 5070 Ti Laptop GPU / CUDA 12.8 上，`Tq=17`、`Tk=257`、`Hq/Hkv=32/4`、`Dh=64`、warmup 1、测量 10 次的本次三阶段 event 均值为 `0.234755 ms`，只作本机计时闭环。

`compute_online()` 让一个 query/head 使用一个线程块，按 32 个 key 分块，在 kernel 内合并最大值、指数和与 Value 累加，不再申请完整 score workspace。它当前只支持 `Dh <= 128`，覆盖 TinyLlama 的 `Dh=64`。每个 128-thread block 包含 4 个 Warp；4 个 Warp 并行计算 4 个 key 的 QK 点积，并在 Warp 内用 shuffle 完成归约，再由各维线程更新 Value 累加。

第一版用整个线程块逐 key 做共享内存树形归约，固定 workload 的三阶段与融合在线均值分别为 `0.224739 ms` 和 `0.296630 ms`，约回退 32%，因此被性能门禁拒绝。改为 4-Warp key 并行后，三轮融合均值为 `0.069920 / 0.069715 / 0.069805 ms`，对应三阶段 `0.223427 / 0.226054 / 0.223434 ms`。这个微基准支持把 v2 标记为 FA2-inspired 候选。

`CudaExecutionContext` 通过 `CudaAttentionPath` 显式选择三阶段或融合路径，默认仍为三阶段。真实 TinyLlama 2 层与 22 层双步对照均通过；22 层 hidden 和 logits 最大差异分别为 `7.63e-06` 和 `9.18e-06`，greedy token 保持 `3561`。这项双步对照固定真实单-query Decode 数值，再由后续 Tensor Prefill 和端到端实验补齐多-query 证据。

CUDA 库只接收 `CudaAttentionPath` 等类型化策略，不读取命令行或环境变量。进程入口负责把 `--attention-path three-stage|fused-online` 转成枚举，再经 Engine Factory 写入请求 Context。CMake 选项决定 CUDA 代码是否参与构建，运行期参数选择已编译实现，benchmark 的 `--batch` 等参数只改变 workload；三者不能互相替代。

随后增加的 `prefill_sequence()` 复用 `[T,D]` 批量算子，并让融合 Attention 一次处理多个 query；TinyLlama 22 层、2-token embedding Prefill 与逐 token 顺序执行的最大误差为 `3.05e-05`。`RealLlamaCudaContext::prefill_tokens()` 再接入批量 embedding、最终 logits 和按 shape 复用的持久 workspace，22 层 logits 最大误差为 `1.19e-05`。连续与分页 CUDA Executor 都使用 Tensor Prefill，真实 Runtime 保持 golden token `3561 12053`。

EXP-GPU-04 进一步比较 32-token Tensor Prefill 和一次 Decode。两轮端到端中位数一轮回退、一轮改善，未形成稳定收益，因而不切换默认路径。融合实现仍有教学价值：它让学习者直接比较完整 score workspace 与在线状态 `(m,l,o)` 的内存行为，也为后续 Nsight 定位和 kernel 调优保留真实入口。

## 分页 Device K/V 与 GQA

`CudaPagedKvCache` 只拥有 `[layer, physical_block, slot, kv_head, head_dim]` 的共享设备 payload；请求 ID、逻辑块顺序和提交状态仍由 Chapter07 `KvBlockPool` 管理。写入接口接收 BlockTable 定位出的物理 block/slot，释放前可按物理块清零。

`paged_gqa_attention()` 把请求的物理块列表上传到可复用 workspace。三阶段 score/value kernel 将逻辑 token 拆为 `logical_block + slot`，经页表取得 physical block 后直接读取 payload，不先拼回连续 K/V。分页融合候选在同一个在线 Softmax kernel 中完成页表定位、QK、状态合并和 Value 累加，不申请完整 score workspace；默认路径是否切换仍由真实 A/B 决定。`prefill_sequence_paged()` 先批量计算一个 chunk 的 Q/K/V，把 K/V 逐行写入页表槽位，再以 chunk 起点作为 `query_position_offset` 执行多 Query 因果 Attention。真实 GPU 测试构造非连续映射，让 chunk 跨过块边界，并与逐词元 Decoder 和 CPU GQA 对照。

EXP-GPU-07 的两轮真实 A/B 中，分页融合 Prefill/Decode 的中位数方向相反，未通过默认路径门禁。Nsight 汇总显示融合 Attention kernel 合计时间约比三阶段少 `32.3%`，但量化 GEMV 占 GPU kernel 时间约 `99.2%`；代码因此保留 `FusedOnline` 显式入口，Runtime 默认继续使用三阶段实现，下一轮优化回到量化 Linear。

`RealLlamaCudaContext` 对分页 Prefill 先预留完整 chunk，所有层成功后才提交 Context 位置和 BlockTable 词元数；失败时恢复 hidden 并裁剪到 chunk 起点。真实双请求 Decode batch 继续验证不同页表共享批量 Linear，并由 Chapter08 统一提交或回滚。

Engine 的 CUDA Factory 用 `PrefixBlockStore` 把同一 `RadixPrefixCache` 算法接到 `KvBlockPool` 和设备 payload。源请求释放时，索引持有的块不会被清空；只有最后一个引用将要释放时，Factory 才在专用 stream 上清理设备块。`RealLlamaCudaContext` 命中后同步推进 position，Executor 通过继续 Prefill 只计算私有后缀。真实 Runtime 测试将该路径与完整冷启动的首个生成词元对照。

## Decoder 逐元素算子

Llama Decoder 使用独立的 RMSNorm 与 SwiGLU CUDA kernel，Residual 继续复用 Vector Add。单向量 RMSNorm 用一个 block 分段归约；批量 RMSNorm 用一行一个 block 处理 `[B,D]`，共享同一 `[D]` 权重。SwiGLU 与 Residual 本身按总元素数执行，可直接处理连续批量 Tensor。测试覆盖非整除宽度 257、TinyLlama hidden size 2048 和多行 CPU golden，这些算子已由下一节的 `CudaDecoderLayer` 组装成合成单层 Decoder。

`CudaExecutionContext::pack_rows()` 与 `unpack_rows()` 在设备内把多个请求的 `[D]` Tensor 和 `[B,D]` 批量 Tensor 相互转换。接口先完成全部 shape 检查再排入 D2D copy；调用方必须先保证各请求 stream 的源数据已经完成。它为共享 RMSNorm/Linear 提供数据交接，不负责请求调度、K/V 所有权或失败回滚。模型创建 Batch Workspace 时按最大上下文预留三阶段 Attention score 和页表临时空间，避免 Decode 位置增长触发逐步扩容。

## 合成单层 Decoder

`CudaDecoderLayer` 按 `RMSNorm -> QKV Linear -> RoPE -> KV append -> GQA -> O Linear -> Residual -> RMSNorm -> gate/up -> SwiGLU -> down -> Residual` 编排已有 backend。权重由 Model 侧只读 Tensor 提供，KV Cache 仍由请求持有，层对象只持有可复用中间 Tensor。

首个测试使用 `D=8、Hq/Hkv=4/2、Dh=2、FFN=12` 的可读合成权重，逐项检查 Attention Norm、GQA context、FFN Norm、SwiGLU 和最终 hidden。它固定整层执行顺序和请求状态语义。

真实权重接入使用 `CudaLinearWeight` 隔离格式差异：Float32 权重按 `[input, output]` Tensor 解释，Q4_K/Q6_K 按 `rows=output、cols=input` 的 packed Matrix 解释。量化 Linear 依据连续输入/输出的元素数校验 shape，因此 Q/K/V 可以直接写入 rank-3 Tensor；单 token 与批量融合路径都不创建完整反量化矩阵。

## 真实层权重上传

`code/model` 中的 `CudaLlamaLayerWeights` 接收 Chapter01 读取的单层权重，把两个 Float32 norm 和七个 Q4_K/Q6_K Linear 上传到显存。对象拥有上传后的 device storage；Chapter01 的 host Tensor 可以在同步完成后释放。上传先写入临时对象，九个权重全部成功后才替换原对象，因此格式错误不会破坏已有 GPU 权重。

`RealLlamaCudaModel` 共享各层 device 权重，`RealLlamaCudaContext` 为每个请求创建 stream、层中间张量，并持有连续 K/V 或绑定分页数据面。一次多层前向只有在所有层成功后才推进 position；失败时已追加的状态会回退到原位置。

`cuda_llama_layer_weights_test` 检查权重方向、packed payload、移动语义和失败原子性。真实模型测试依次核对 layer-0、2/22 层 hidden 与 K/V、量化 embedding、final norm、LM Head 和 logits；Generation 测试继续验证真实 tokenizer、greedy decode、detokenizer 与请求隔离。`CudaDecoderBatchLayer` 合并无状态计算，并让每个请求通过连续缓存或页表读取自己的历史。真实 Runtime 和 Service 测试检查 logits、golden token、预留/提交、调度计数、整批回滚、终态释放和 workspace 复用。

## 运行与验证

当前验证工具链是 CUDA 12.8。脚本默认使用 `/usr/local/cuda-12.8`：

```bash
bash code/cuda/scripts/run.sh
```

继续验证真实 TinyLlama layer-0：

```bash
bash code/engine/scripts/run_real_cuda_first_layer.sh
```

验证 2 层和完整 22 层 Decoder：

```bash
bash code/engine/scripts/run_real_cuda_decoder.sh
```

验证固定 token prompt 的 Generation Runtime：

```bash
bash code/engine/scripts/run_real_cuda_runtime.sh
```

验证真实文本生成：

```bash
bash code/engine/scripts/run_real_cuda_generation.sh
```

运行环境必须能访问 `/dev/nvidia*`。受控沙箱隐藏设备节点时，需要在宿主环境执行；编译通过不能代替真实 CTest。
