# CUDA 公共基础层

`code/cuda` 为后续 CUDA backend 提供独立基础库，不让 CPU 默认回归依赖 CUDA Toolkit。当前已完成资源层、Linear、RoPE、连续 KV、causal GQA，以及单层组装所需的 RMSNorm、Residual 和 SwiGLU 算子。

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
| `cuda_execution_context_test.cu` | Tensor 复制、算子调度、workspace 复用和移动测试 |
| `cuda_quantized_matrix.h/.cpp` | packed Q4_K/Q6_K row-major device 权重 |
| `quantized_dequantize_backend.h/.cu` | Q4_K/Q6_K 到 Float32 workspace 的 CUDA 解码 |
| `cuda_quantized_linear_test.cu` | CPU reference、dequantize + SGEMM、错误路径和固定 workload 计时 |
| `core/rope_contract.h` | 固定原地 RoPE 的连续布局、旋转维度、位置和频率参数 |
| `rope_backend.h/.cu` | 每线程处理一个相邻分量对的 CUDA RoPE backend |
| `cuda_rope_test.cu` | 多 token/head CPU golden、局部旋转、错误参数和 event 计时 |
| `cuda_kv_cache.h/.cpp` | 请求级连续 K/V device storage、D2D append、clear 和显式释放 |
| `cuda_kv_cache_test.cu` | append 偏移、容量、shape、移动、复用和请求隔离 |
| `core/gqa_attention_contract.h` | 固定 Q/KV head 映射、causal position、scale 和布局 |
| `gqa_attention_backend.h/.cu` | score、稳定 softmax、value aggregation 三阶段 CUDA backend |
| `cuda_gqa_attention_test.cu` | prefill/decode CPU golden、错误路径和 event 计时 |
| `core/decoder_elementwise_contract.h` | RMSNorm 与 SwiGLU 连续向量契约 |
| `decoder_elementwise_backend.h/.cu` | 单 block RMSNorm 和逐元素 SwiGLU kernel |
| `cuda_decoder_elementwise_test.cu` | 257/2048 宽度 CPU golden 与错误 shape |
| `cuda_decoder_layer.h/.cpp` | 单 token Llama Decoder 的算子编排与中间 Tensor 所有权 |
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

## Quantized Linear 基线

`CudaQuantizedMatrix` 保存 packed Q4_K/Q6_K 权重，`CudaExecutionContext::quantized_linear()` 先将权重解码到复用的 Float32 Workspace，再调用现有 cuBLAS SGEMM。该路径刻意保留完整中间矩阵，用于核对 block 映射、shape、字节数和错误传播；它不是最终 fused kernel。

固定 workload 使用 `rows=256`、`cols=256`、warmup 1 次和测量 10 次。计时在权重与输入上传后开始，包含每轮 CUDA dequantize kernel、cuBLAS SGEMM、主机提交以及末尾同步，不包含 H2D/D2H；未来 fused A/B 必须复用相同定义。

## RoPE Kernel

RoPE 请求使用连续 `[tokens, heads, head_dim]` 布局，在原 Tensor 上旋转前 `rotary_dimension` 个分量。相邻两个分量组成一对，每个 CUDA 线程负责一对；`position_offset + token_index` 让 prefill 与后续 decode 使用同一位置语义。`rotary_dimension` 可以小于 `head_dim`，未参与旋转的尾部保持不变。

测试先用同构 CPU 公式生成 golden，再覆盖多 token、多 head、非整除 grid、局部旋转维度、非法参数和 `CudaExecutionContext` shape 检查。RTX 5070 Ti Laptop GPU / CUDA 12.8 上，`257 x 32 x 64`、warmup 1 次、测量 10 次的本次 kernel event 均值为 `0.0083808 ms`；该数字只用于确认计时闭环，不是跨设备性能基线。

## 连续 Device KV Cache

`CudaKvCache` 分别独占 `[capacity_tokens, kv_heads, head_dim]` 的 K/V 连续显存。新 K/V 使用相同布局的 `CudaTensor` 输入，并在调用方 stream 上追加到当前 token 尾部；成功后才推进逻辑长度。`clear()` 只把长度归零以复用 allocation，`reset()` 才释放显存。

首版保持连续布局，用于固定 prefill/decode 的数据语义和请求所有权。每个请求、每层应持有独立缓存；分页 block、回收池和 prefix sharing 属于后续 Runtime/服务优化，不能混入 P1-3 的数值基线。

## Causal GQA Attention

`CudaGqaAttentionBackend` 接收 Q `[Tq,Hq,Dh]` 和连续缓存 K/V `[Tk,Hkv,Dh]`。query head 按 `query_head / (Hq/Hkv)` 共享 KV head；`query_position_offset + query_token` 决定 causal mask 的最后可见位置。`CudaExecutionContext::append_kv()`、`gqa_attention()` 保证上传、D2D append 和计算进入同一 stream。

首版刻意保留三个 kernel：先生成 `[Tq,Hq,Tk]` score workspace，再逐行执行稳定 softmax，最后聚合 value。该路径用于核对 prefill/decode 和 GQA 数值，不是 FlashAttention。RTX 5070 Ti Laptop GPU / CUDA 12.8 上，`Tq=17`、`Tk=257`、`Hq/Hkv=32/4`、`Dh=64`、warmup 1、测量 10 次的本次三阶段 event 均值为 `0.234755 ms`，只作本机计时闭环。

## Decoder 逐元素算子

Llama Decoder 使用独立的 RMSNorm 与 SwiGLU CUDA kernel，Residual 继续复用 Vector Add。RMSNorm 单 block 分段累加平方和，再按 `input * rsqrt(mean_square + epsilon) * weight` 写回；SwiGLU 按 `silu(gate) * up` 逐元素计算。测试覆盖非整除宽度 257 和 TinyLlama hidden size 2048，这些算子已由下一节的 `CudaDecoderLayer` 组装成合成单层 Decoder。

## 合成单层 Decoder

`CudaDecoderLayer` 按 `RMSNorm -> QKV Linear -> RoPE -> KV append -> GQA -> O Linear -> Residual -> RMSNorm -> gate/up -> SwiGLU -> down -> Residual` 编排已有 backend。权重由 Model 侧只读 Tensor 提供，KV Cache 仍由请求持有，层对象只持有可复用中间 Tensor。

首个测试使用 `D=8、Hq/Hkv=4/2、Dh=2、FFN=12` 的可读合成权重，逐项检查 Attention Norm、GQA context、FFN Norm、SwiGLU 和最终 hidden。它证明整层执行顺序和所有权已闭合，不代表真实 TinyLlama 量化首层已经接入。

真实权重接入使用 `CudaLinearWeight` 隔离格式差异：Float32 权重按 `[input, output]` Tensor 解释，Q4_K/Q6_K 按 `rows=output、cols=input` 的 packed Matrix 解释。量化 Linear 依据连续输入/输出的元素数校验 shape，因此 Q/K/V 可以直接写入 rank-3 Tensor；workspace 中的反量化矩阵只在计算期间复用，不成为永久模型副本。

## 运行与验证

当前验证工具链是 CUDA 12.8。脚本默认使用 `/usr/local/cuda-12.8`：

```bash
bash code/cuda/scripts/run.sh
```

运行环境必须能访问 `/dev/nvidia*`。受控沙箱隐藏设备节点时，需要在宿主环境执行；编译通过不能代替真实 CTest。
