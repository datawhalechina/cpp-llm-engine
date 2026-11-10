# 推理引擎公式索引

公式索引统一教程中的符号、shape 和计算口径。章节正文负责解释直觉和实现过程；公式索引负责给出稳定编号、输入输出、代码位置与验证方式。

第一版使用 Markdown 代码块，不依赖额外数学渲染插件。公式编号是跨章节链接的稳定标识，修改表达时不更换已有编号。

## 通用符号

阅读公式前先固定 batch、序列、hidden、head 和矩阵维度的写法。

| 符号 | 含义 | 常见 shape |
| --- | --- | --- |
| `B` | batch size | 请求数或序列数 |
| `T` | 当前 token 数 | prompt 长度或 decode step 长度 |
| `D` | hidden size | TinyLlama 为 2048 |
| `Hq` | query head 数 | GQA 中可大于 KV head 数 |
| `Hkv` | key/value head 数 | `Hq` 必须能被 `Hkv` 整除 |
| `Dh` | head dimension | 通常为 `D / Hq` |
| `M/N/K` | MatMul 三个逻辑维度 | `[M,K] × [K,N] -> [M,N]` |
| `L` | Decoder layer 数 | TinyLlama 为 22 |
| `S` | 已缓存的序列长度 | KV Cache 当前 token 数 |

## 模型文件与加载

<a id="f-load-01"></a>
### F-LOAD-01 Payload 对齐与张量范围

GGUF metadata 和 tensor index 结束后，权重 payload 从满足 alignment 的下一个地址开始。读取张量前必须同时验证相对 offset 和字节范围。

```text
data_offset = ceil(index_end / alignment) × alignment
tensor_begin = data_offset + tensor_offset
tensor_end = tensor_begin + tensor_bytes

required: tensor_begin >= data_offset
required: tensor_end <= file_size
```

代码对应：Chapter01 `GgufReader`、`GgufTensorLoader`。验证重点：alignment、offset 加法溢出、截断文件和 tensor payload 越界。

## Tensor 与布局

<a id="f-tensor-01"></a>
### F-TENSOR-01 Numel

Tensor 元素数由所有维度相乘得到；任何一维为 0 时，是否允许空 Tensor 由具体 contract 决定。

```text
numel(shape) = product(shape[i]), i = 0 .. rank-1
bytes = numel × bytes_per_element(dtype)
```

代码对应：Chapter02 `Tensor::numel()`、core `TensorContract`。验证重点：空 shape、零维、乘法溢出和 dtype 字节数。

<a id="f-tensor-02"></a>
### F-TENSOR-02 连续 Stride 与 View Offset

行主序连续 Tensor 的最后一维 stride 为 1，前一维 stride 等于后一维 stride 与后一维大小的乘积。

```text
stride[rank - 1] = 1
stride[i] = stride[i + 1] × shape[i + 1]

storage_index = offset + sum(index[i] × stride[i])
```

代码对应：Chapter02 N 维索引、reshape、slice、transpose 和共享 Storage。验证重点：view 不复制、offset/stride 正确、越界拒绝。

## Linear 与复杂度

<a id="f-linear-01"></a>
### F-LINEAR-01 Linear / MatMul

教程统一把一批输入行乘以权重转置，bias 按输出列广播。

```text
X: [M, K]
W: [N, K]
b: [N]

Y = X × W^T + b
Y: [M, N]

multiply-add complexity = O(M × N × K)
```

代码对应：Chapter03 MatMul、Chapter04 Linear、Chapter05 Quantized Linear 和 CUDA cuBLAS backend。验证重点：shape、row-major 映射、CPU golden 和误差容限。

## Normalization

<a id="f-norm-01"></a>
### F-NORM-01 LayerNorm

LayerNorm 先减均值，再按方差归一化，最后应用逐元素 scale 和 bias。

```text
mean = sum(x[i]) / D
variance = sum((x[i] - mean)^2) / D
y[i] = (x[i] - mean) / sqrt(variance + epsilon) × gamma[i] + beta[i]
```

代码对应：Chapter04 教学 `LayerNorm`。验证重点：常量输入、epsilon、有限值和逐维 golden。

<a id="f-norm-02"></a>
### F-NORM-02 RMSNorm

RMSNorm 不减均值，只按均方根缩放。TinyLlama Decoder 使用 RMSNorm。

```text
rms = sqrt(sum(x[i]^2) / D + epsilon)
y[i] = x[i] / rms × weight[i]
```

代码对应：`code/model` 中 `RealLlamaCpuModel` 的 Attention/FFN/Final RMSNorm，由 Chapter04 讲解。验证重点：22 层有限值、最终 logits 和 llama.cpp 数值对照。

## Attention、GQA 与 RoPE

<a id="f-attn-01"></a>
### F-ATTN-01 Q/K/V Projection

hidden state 通过三个 Linear 生成 query、key 和 value。GQA 的 K/V 输出宽度小于 Q 输出宽度。

```text
Q = X × Wq^T
K = X × Wk^T
V = X × Wv^T

Q: [T, Hq, Dh] or [B, T, Hq, Dh]
K: [T, Hkv, Dh] or [B, T, Hkv, Dh]
V: [T, Hkv, Dh] or [B, T, Hkv, Dh]
```

代码对应：Chapter04 Attention 和 Chapter01 真实 TinyLlama Decoder。

<a id="f-attn-02"></a>
### F-ATTN-02 Scaled Dot-Product Attention

每个 query 与可见 key 计算缩放分数，causal mask 屏蔽未来位置，softmax 权重再聚合 value。

```text
scores = Q × K^T / sqrt(Dh)
probabilities = softmax(scores + causal_mask)
context = probabilities × V
```

代码对应：Chapter04 causal Attention。验证重点：未来 token 不可见、softmax 行和为 1、prefill/decode 输出一致。

<a id="f-attn-03"></a>
### F-ATTN-03 GQA Head 映射

Grouped-Query Attention 让多个 query head 共享一个 KV head。

```text
group_size = Hq / Hkv
kv_head(query_head) = floor(query_head / group_size)

required: Hq % Hkv == 0
```

代码对应：Chapter01 TinyLlama GQA 与 `code/cuda` 三阶段 GQA backend。验证重点：head 边界、K/V width、每组映射和 prefill/decode 一致性。

<a id="f-attn-04"></a>
### F-ATTN-04 分块在线 Softmax

每个分块保存最大值 `m_b`、指数和 `l_b` 和未归一化 Value 累加 `o_b`。合并时先切换到共同指数基准，避免直接对大分数取指数。

```text
m_new = max(m, m_b)
alpha = exp(m - m_new)
beta  = exp(m_b - m_new)

l_new = alpha * l + beta * l_b
o_new = alpha * o + beta * o_b
output = o / l
```

代码对应：Chapter04 CPU 分块在线 Attention 参考路径。验证重点：不同分块大小、极端分数、因果可见范围和三段式参考输出一致。

<a id="f-rope-01"></a>
### F-ROPE-01 Rotary Position Embedding

RoPE 对 Q/K 的相邻二维分量做位置相关旋转，不直接修改 V。

```text
theta(i, position) = position / base^(2i / Dh)

y[2i]     = x[2i] × cos(theta) - x[2i+1] × sin(theta)
y[2i + 1] = x[2i] × sin(theta) + x[2i+1] × cos(theta)
```

代码对应：Chapter06 `position_encoding` 和 Chapter01 真实 Decoder。验证重点：position 0、不同位置、Q/K 同规则和 rotary dimension。

## FFN

<a id="f-ffn-01"></a>
### F-FFN-01 SwiGLU

Llama 系列 FFN 使用 gate、up 和 down 三个 Linear；gate 经 SiLU 后与 up 逐元素相乘。

```text
SiLU(x) = x / (1 + exp(-x))
hidden = SiLU(X × W_gate^T) elementwise_mul (X × W_up^T)
Y = hidden × W_down^T
```

代码对应：Chapter01 真实 TinyLlama FFN。Chapter04 教学 FFN 使用 ReLU，不能把两者写成同一个实现。

## KV Cache 与内存

<a id="f-kv-01"></a>
### F-KV-01 KV Cache Payload

单请求连续 KV Cache 同时保存每层的 key 和 value。

```text
elements = L × 2 × S × Hkv × Dh
bytes = elements × bytes_per_element
```

代码对应：Model `RealLlamaCpuContext`、Chapter04 连续 KV，以及 Chapter07 `CpuPagedKvCache`。Block KV 改变物理分配与寻址，不改变逻辑 K/V 元素数。

<a id="f-sched-01"></a>
### F-SCHED-01 调度预算与 KV Block 数

调度器必须同时满足 token、请求调度槽位和 KV block 三类容量。这里的请求槽位表示本 step 可接纳的活跃请求数，不是物理 K/V 块中的 token 槽位。连续序列映射到固定大小 block 时，最后一个 block 可以只使用一部分。

```text
scheduled_tokens = sum(tokens_selected_for_request[i])
required_blocks(tokens) = ceil(tokens / tokens_per_block)

required: scheduled_tokens <= step_token_budget
required: scheduled_requests <= available_request_slots
required: sum(required_blocks[i]) <= free_blocks
```

代码对应：Chapter07 `KvBlockPool`、Chapter08 `StepScheduler` 和 Chapter10 `ServiceCore::dispatch_step()`。验证重点：预算不足不执行、整除边界、终态释放和 block 复用。

## Quantization

<a id="f-quant-01"></a>
### F-QUANT-01 反量化

对称量化通常省略 zero point；非对称量化同时保存 scale 和 zero point。GGUF K-quant 还会在 block 内保存分级 scale，实际布局以格式 contract 为准。

```text
symmetric:  x_approx = scale × q
asymmetric: x_approx = scale × (q - zero_point)

Q8_0: x_approx = block_scale × signed_q8
Q4_K: x_approx = block_scale × group_scale × q4
                   - block_min × group_min
Q6_K: x_approx = block_scale × group_scale × signed_q6
```

这些是解释参数层级的概念公式；实际实现还必须按格式完成 half 转换、bit unpack 和分组索引。代码对应：Chapter05 Q4_K/Q6_K/Q8_0 reference、fused CPU kernel 和 CUDA dequantize backend。

<a id="f-quant-02"></a>
### F-QUANT-02 压缩率与误差

压缩率必须使用实际 block 字节数，不能只按“4 bit”或“6 bit”估算，因为 block 还包含 scale 和元数据。

```text
compression_ratio = float_weight_bytes / quantized_payload_bytes
absolute_error = abs(reference - actual)
relative_error = absolute_error / max(abs(reference), epsilon)
```

代码对应：Chapter05 block contract 和 correctness 测试。

<a id="f-quant-03"></a>
### F-QUANT-03 动态激活量化

动态对称量化先从当前激活计算 scale，再让低精度矩阵 kernel 消费整数输入。实际实现可以按 Tensor、row 或 token 选择更细粒度。

```text
activation_scale = max(abs(x)) / qmax
q_x = clamp(round(x / activation_scale), -qmax, qmax)

W8A8 accumulation:
y_approx = weight_scale × activation_scale × sum(q_w × q_x)
```

当前 Chapter05 尚未实现该路径；接入时需要独立验证 scale 统计、饱和比例、Linear 输出和最终 logits。

<a id="f-quant-04"></a>
### F-QUANT-04 K/V Cache 量化

K/V 可以使用不同 scale，并按层、head、token 或 block 保存 metadata。下面只表达对称量化语义，不规定生产布局。

```text
q_K = clamp(round(K / scale_K), -qmax, qmax)
q_V = clamp(round(V / scale_V), -qmax, qmax)

K_approx = scale_K × q_K
V_approx = scale_V × q_V
```

Chapter07 当前保存浮点 K/V；未来量化实现仍需保持 BlockTable、提交、回滚和引用计数语义。

## CUDA 执行

<a id="f-cuda-01"></a>
### F-CUDA-01 一维 Grid 配置与边界保护

一维 kernel 通常按 block size 向上取整得到 grid size。多出的线程必须通过索引边界检查退出。

```text
grid_size = ceil(element_count / block_size)
global_index = block_index × block_size + thread_index

if global_index < element_count:
    process(global_index)
```

代码对应：`code/cuda` 的 vector add、dequantize 和量化 Linear baseline。验证重点：非 block 整数倍长度、零长度策略、CPU/CUDA 数值一致和 stream 同步位置。

## Sampling

<a id="f-sample-01"></a>
### F-SAMPLE-01 Temperature Softmax 与 Greedy

temperature 改变 logits 的相对尖锐程度；Greedy 直接选择最大 logit，不需要随机采样。

```text
p[i] = exp(logit[i] / temperature) / sum(exp(logit[j] / temperature))
greedy_token = argmax(logit)
```

代码对应：Chapter06 当前可执行实现为 Greedy；Temperature、Top-K 和 Top-P 仍是扩展项。

## 数值评估

<a id="f-eval-01"></a>
### F-EVAL-01 余弦相似度

对长度相同的两个向量 `x` 和 `y`：

```text
cosine_similarity(x, y) = dot(x, y) / (norm(x) * norm(y))
```

结果接近 `1` 表示两个向量方向相近，接近 `0` 表示整体变化方向缺少相关性，接近 `-1` 表示方向相反。它对整体缩放不敏感，因此不能替代最大误差、平均绝对误差、Top-K 重合或贪心词元检查。教程用它比较 hidden state 和 logits 的整体数值结构。

## Benchmark 与 Serving

<a id="f-perf-01"></a>
### F-PERF-01 请求延迟

时间点必须来自同一时钟。TPOT 只统计首 token 之后的生成间隔。

```text
queue_wait = dispatch_time - accepted_time
TTFT = first_token_time - accepted_time
completion_latency = finished_time - accepted_time
TPOT = (finished_time - first_token_time) / max(generated_tokens - 1, 1)
```

代码对应：Chapter09 性能证据、Chapter10 `ServiceMetrics`。

<a id="f-perf-02"></a>
### F-PERF-02 吞吐与加速比

吞吐的分子和时间区间必须明确；跨机器结果不能直接计算加速比。

```text
token_throughput = completed_tokens / measured_seconds
request_throughput = completed_requests / measured_seconds
speedup = baseline_time / optimized_time
```

代码对应：Chapter09 多请求 benchmark、Profiling 和 A/B 证据。

<a id="f-serve-01"></a>
### F-SERVE-01 请求计数与终态守恒

服务指标是请求生命周期的可观测投影。若统计窗口内没有跨窗口遗留请求，接受数应等于各终态数量之和，且在途请求归零。

```text
terminal = completed + failed + cancelled + timed_out
inflight = accepted - terminal

closed-window required: accepted = terminal
closed-window required: inflight = 0
resource required: released = terminal
```

代码对应：Chapter10 `ServiceMetrics`、`code/engine` 的嵌入请求服务测试。验证重点：成功、异常、客户端断开、超时和 Context 生命周期释放。

## 多模态

<a id="f-mm-01"></a>
### F-MM-01 Image Token 到 Pixel Budget

IDEFICS3 的 merge size 表示每边合并多少 patch，因此一个输出 image token 覆盖 `(patch_size × merge_size)^2` 个输入像素。

```text
patch_area_per_image_token = (patch_size × merge_size)^2
pixel_budget = image_token_budget × patch_area_per_image_token

SmolVLM baseline:
(16 × 4)^2 = 4096 pixels/token
256 × 4096 = 1,048,576 pixels
```

代码对应：`code/multimodal` 的 `Idefics3TilingConfig` 和 layout 计算。

<a id="f-mm-02"></a>
### F-MM-02 Patch、Tile 与视觉图片数

每张 tile 的 patch 数由 tile 和 patch 的边长决定；动态切图额外加入一张 overview。

```text
patches_per_tile = (tile_width / patch_width) × (tile_height / patch_height)
tile_count = grid_x × grid_y
visual_image_count = 1 + tile_count

SmolVLM baseline:
patches_per_tile = 32 × 32 = 1024
visual_image_count = 1 + 3 × 2 = 7
```

代码对应：`extract_patches()`、`Idefics3Layout::visual_image_count()` 和多模态 inspect。

<a id="f-mm-03"></a>
### F-MM-03 文本与视觉 Embedding 拼接

多模态 Runtime 消费统一宽度的 embedding 序列。视觉占位区替换为 Projector 输出，前后文本 token 保持原顺序。

```text
text_prefix: [T_prefix, D]
visual:      [T_visual, D]
text_suffix: [T_suffix, D]

sequence = concat(text_prefix, visual, text_suffix, axis=token)
sequence_shape = [T_prefix + T_visual + T_suffix, D]
```

代码对应：`code/multimodal` 的 prompt composition、Projector 和 `EmbeddingRequestExecutor`。验证重点：placeholder 区间、hidden width、拼接顺序、同步/SSE token 一致和终态释放。

## 使用规则

1. 章节第一次使用公式时解释直觉，并链接公式编号。
2. shape、单位或时间区间变化时，必须显式写出，不能沿用同一编号掩盖语义变化。
3. 公式实现必须有数值测试、shape 测试或 benchmark 字段作为验证证据。
4. 新公式先加入索引并分配编号，再在多个章节中引用。
