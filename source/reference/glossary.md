# 推理引擎术语表

术语表统一教程中的中文名称、英文名称、缩写和使用范围。章节第一次出现术语时仍需解释上下文；术语表提供稳定查询入口，不替代章节正文。

## Tensor、内存与执行

这些术语描述数据怎样保存、共享和移动，以及算子在哪个设备上执行。

| 中文术语 | 英文或缩写 | 定义 | 首次学习 | 相关公式 |
| --- | --- | --- | --- | --- |
| 张量 | Tensor | 带 shape、dtype、device 和布局语义的数据对象 | Chapter01-02 | [F-TENSOR-01](./formulas.md#f-tensor-01) |
| 形状 | shape | Tensor 每个维度的长度 | Chapter01 | [F-TENSOR-01](./formulas.md#f-tensor-01) |
| 维数 | rank | Tensor 的轴数量，也就是 shape 中维度的个数 | Chapter02 | [F-TENSOR-01](./formulas.md#f-tensor-01) |
| 轴 | axis | shape、stride 和索引中的某一个维度位置 | Chapter02 | [F-TENSOR-02](./formulas.md#f-tensor-02) |
| 步长 | stride | 索引在某一维增加 1 时跨过的 storage 元素数 | Chapter02 | [F-TENSOR-02](./formulas.md#f-tensor-02) |
| 偏移 | offset | 当前 view 的第一个元素在底层 storage 中的位置 | [Chapter02 Slice](../chapter02_tensor_ops/03_slice.md) | [F-TENSOR-02](./formulas.md#f-tensor-02) |
| 元素总数 | number of elements, numel | shape 各维长度的乘积；reshape 前后必须保持不变 | [Chapter02 Reshape](../chapter02_tensor_ops/02_reshape.md) | [F-TENSOR-01](./formulas.md#f-tensor-01) |
| 底层存储 | Storage | 保存实际字节并可被多个 Tensor view 共享的对象 | Chapter02 | [F-TENSOR-02](./formulas.md#f-tensor-02) |
| 视图 | view | 共享 Storage、只改变 shape/stride/offset 的 Tensor | Chapter02 | [F-TENSOR-02](./formulas.md#f-tensor-02) |
| 连续布局 | contiguous layout | 元素按约定维度顺序连续排列的内存布局 | Chapter02 | [F-TENSOR-02](./formulas.md#f-tensor-02) |
| 工作区 | workspace | 算子执行期间复用的临时内存 | Chapter03、GPU 路线 | [F-TENSOR-01](./formulas.md#f-tensor-01) |
| 算子 | Operator | 对调用方提供稳定输入输出约定、把具体计算交给后端执行的接口 | [Chapter03 Operator Base](../chapter03_matmul/01_operator_base.md) | [F-LINEAR-01](./formulas.md#f-linear-01) |
| 后端 | Backend | 在 CPU、CUDA 或外部库上实现同一算子 contract 的执行组件 | Chapter03 | [F-LINEAR-01](./formulas.md#f-linear-01) |
| 内核 | Kernel | 完成具体数值循环或 GPU 并行计算的底层实现 | Chapter03、Foundation CUDA | [F-LINEAR-01](./formulas.md#f-linear-01) |
| 执行上下文 | Execution Context | 持有请求状态或设备 stream、handle、workspace 的执行对象 | Chapter06、GPU 路线 | [F-KV-01](./formulas.md#f-kv-01) |
| 主机内存 | host memory | CPU 可直接访问的内存；GPU kernel 通常不能把普通 host pointer 当作 device pointer 使用 | Foundation CUDA、GPU 路线 | - |
| 设备内存 | device memory | GPU kernel 可访问、由设备 allocator 管理的内存 | Foundation CUDA、GPU 路线 | [F-CUDA-01](./formulas.md#f-cuda-01) |
| CUDA 流 | CUDA stream | 保持队列内顺序、允许不同队列并发的异步 GPU 工作序列 | GPU 路线 | [F-CUDA-01](./formulas.md#f-cuda-01) |
| CUDA 事件 | CUDA event | 记录 stream 进度并用于依赖同步或设备侧计时的对象 | GPU 路线 | - |
| 进程 | process | 拥有独立地址空间的程序执行实例；跨进程对象需要 IPC、共享内存或网络传递 | Foundation / Chapter10 | - |
| 线程 | thread | 进程内的执行单元；共享地址空间，因此并发修改对象需要所有权或同步规则 | Foundation / Chapter10 | - |
| Warp | warp | CUDA 中按同一指令协同执行的一组线程；当前 NVIDIA GPU 通常每组 32 个线程 | Chapter05 | `threadIdx.x / 32` |
| 归约 | reduction | 把多个线程的局部结果按求和、最大值等规则合并为一个结果 | Chapter05 / Chapter04 | `warp_reduce` |
| 合并内存访问 | coalesced memory access | 相邻线程访问相邻 device 地址，使多个请求尽量合并为较少的显存事务 | Chapter05 | packed weights / input |
| 同步调用 | synchronous call | 调用返回时，调用者需要的结果或错误已经可见 | Foundation / Chapter06、10 | - |
| 异步工作 | asynchronous work | 先提交工作，完成状态稍后通过 callback、future、event 或队列出现 | Foundation / CUDA / Chapter08、10 | - |

## 模型与算子

这些术语对应 Transformer Decoder 和 Vision Transformer 共同使用的数值结构。

| 中文术语 | 英文或缩写 | 定义 | 首次学习 | 相关公式 |
| --- | --- | --- | --- | --- |
| 模型加载器 | Model Loader | 将模型 metadata、tensor index 和 payload 转成运行时模型对象 | Chapter01 | - |
| 分词器 | Tokenizer | 在文本与 token ID 序列之间转换，并执行模型规定的归一化、切分和特殊 token 规则 | Chapter00-01 | - |
| 词元编号 | token ID | 词元在模型词表中的整数编号，是模型实际接收和输出的离散值 | Chapter00-01 | - |
| 词表 | vocabulary | token id、token 字节序列及分词属性的映射集合 | Chapter01 | - |
| 模型元数据 | model metadata | 描述模型架构、shape、tokenizer 和量化参数的键值信息 | Chapter01 | [F-LOAD-01](./formulas.md#f-load-01) |
| 张量索引 | tensor index | 记录权重名称、shape、dtype 和 payload offset 的目录 | Chapter01 | [F-LOAD-01](./formulas.md#f-load-01) |
| 数据对齐 | alignment | 让 payload 起始位置满足固定字节倍数，便于映射和设备访问 | Chapter01 | [F-LOAD-01](./formulas.md#f-load-01) |
| 矩阵乘法 | MatMul | 将 `[M,K]` 与 `[K,N]` 相乘得到 `[M,N]` | Chapter03 | [F-LINEAR-01](./formulas.md#f-linear-01) |
| 线性层 | Linear | MatMul 加可选 bias 的模型层 | Chapter03-04 | [F-LINEAR-01](./formulas.md#f-linear-01) |
| 注意力 | Attention | 根据 Q/K 相关性对 V 加权聚合 | Chapter04 | [F-ATTN-02](./formulas.md#f-attn-02) |
| 多头注意力 | Multi-head Attention, MHA | 将隐藏维度拆成多个 head 分别计算 Attention，再合并回原维度 | [Chapter04 Multi-head Attention](../chapter04_attention/01_multihead_attention.md) | [F-ATTN-02](./formulas.md#f-attn-02) |
| 分组查询注意力 | Grouped-Query Attention, GQA | 多个 query head 共享较少的 KV head | Chapter04 | [F-ATTN-03](./formulas.md#f-attn-03) |
| 因果掩码 | causal mask | 阻止当前位置读取未来 token 的 Attention mask | Chapter04 | [F-ATTN-02](./formulas.md#f-attn-02) |
| 层归一化 | LayerNorm | 使用均值和方差归一化 hidden state | Chapter04 | [F-NORM-01](./formulas.md#f-norm-01) |
| 均方根归一化 | RMSNorm | 不减均值、按均方根缩放 hidden state | Chapter04、真实 Decoder | [F-NORM-02](./formulas.md#f-norm-02) |
| 旋转位置编码 | Rotary Position Embedding, RoPE | 对 Q/K 二维分量施加位置相关旋转 | Chapter04-05 | [F-ROPE-01](./formulas.md#f-rope-01) |
| 前馈网络 | Feed-Forward Network, FFN | Transformer block 中 Attention 后的非线性变换 | Chapter04 | [F-FFN-01](./formulas.md#f-ffn-01) |
| 门控激活 | SwiGLU | 使用 SiLU gate 与 up 分支逐元素相乘的 FFN 结构 | Chapter04、真实 Decoder | [F-FFN-01](./formulas.md#f-ffn-01) |
| 解码器 | Decoder | 组合 Attention、FFN、Residual 和 Norm 的生成模型主体 | Chapter04 | [F-ATTN-02](./formulas.md#f-attn-02) |
| 残差连接 | residual connection | 将子层输出与输入相加，形成 Transformer block 的主数据通路 | Chapter04 | - |
| 候选分数 | logits | 模型对词表中每个候选 token 输出的未归一化分数 | Chapter04-06 | [F-SAMPLE-01](./formulas.md#f-sample-01) |

## 生成、缓存与调度

这些术语描述一个请求从 prompt 到逐 token 输出的状态变化。

| 中文术语 | 英文或缩写 | 定义 | 首次学习 | 相关公式 |
| --- | --- | --- | --- | --- |
| 词元 | token | tokenizer 和模型处理的离散输入输出单位 | Chapter00-01 | - |
| 提示输入 | prompt | 用户提供并在生成前完成 prefill 的输入序列 | Chapter00、Chapter06 | - |
| 请求 | Request | 保存 prompt、生成 token、执行阶段、停止原因和请求级资源的运行时对象 | [Chapter06 Request 生命周期](../chapter06_generation/next/04_request_lifecycle.md) | [F-SCHED-01](./formulas.md#f-sched-01) |
| 上下文窗口 | context window | 模型一次请求最多能够处理的词元范围，描述容量限制，不是一个 C++ 状态对象 | Chapter00、Chapter06 | - |
| 模型执行上下文 | model execution context, Context | 保存或绑定一个请求的当前位置、分层 K/V 和后端执行状态；不保存所有请求共享的模型权重 | [Chapter04 解码器层](../chapter04_attention/04_decoder_layer.md) | [F-KV-01](./formulas.md#f-kv-01) |
| 请求模型执行器 | Request Model Executor, Executor | 接收一个请求的 prefill/decode 输入，使用该请求的 Context 返回 logits | [Chapter06 请求与执行契约](../chapter06_generation/next/01_request_contract.md) | - |
| 执行器工厂 | Executor Factory, Factory | 集中持有共享模型和后端资源，并按统一规则创建请求执行器及其 Context 的长生命周期对象 | [Chapter06 请求与执行契约](../chapter06_generation/next/01_request_contract.md) | - |
| 预填充 | prefill | 批量处理 prompt 并建立初始 KV Cache 的阶段 | Chapter06 | [F-KV-01](./formulas.md#f-kv-01) |
| 逐词元解码 | decode | 利用已有 KV Cache 每步生成一个新词元的阶段 | Chapter06 | [F-KV-01](./formulas.md#f-kv-01) |
| 键值缓存 | Key-Value Cache, KV Cache | 保存历史 token 的 Attention key/value tensor | Chapter04、Chapter07 | [F-KV-01](./formulas.md#f-kv-01) |
| 分块预填充 | chunked prefill | 按词元预算将长输入分多个调度步执行 | Chapter08 | [F-KV-01](./formulas.md#f-kv-01) |
| 连续批处理 | continuous batching | 每个调度 step 按请求状态重新组成执行 batch | Chapter08 | [F-PERF-02](./formulas.md#f-perf-02) |
| 词元预算 | token budget | 单个调度步允许处理的词元上限 | Chapter08 | [F-SCHED-01](./formulas.md#f-sched-01) |
| KV 块池 | KV block pool | 按固定大小 block 分配、复用和回收 KV Cache 容量的资源池 | Chapter07 | [F-SCHED-01](./formulas.md#f-sched-01) |
| 分页注意力 | PagedAttention | 通过逻辑 block 到物理 block 的映射读取非连续 KV Cache 的注意力方法 | Chapter07 | [F-KV-01](./formulas.md#f-kv-01) |
| 采样器 | Sampler | 根据 logits 选择下一 token 的组件 | Chapter06 | [F-SAMPLE-01](./formulas.md#f-sample-01) |
| 贪心采样 | Greedy Sampling | 每步选择最大 logit 对应 token | Chapter06 | [F-SAMPLE-01](./formulas.md#f-sample-01) |
| 运行时 | Runtime | 组织模型执行、缓存、采样和请求状态推进的组件 | Chapter06 | - |
| 调度器 | Scheduler | 根据 slot、token 和内存预算选择本 step 请求 | Chapter08 | - |
| 流式输出 | streaming | token 生成后立即向调用方持续返回事件 | Chapter06、Chapter10 | [F-PERF-01](./formulas.md#f-perf-01) |

## 量化、性能与服务

这些术语用于区分数据压缩、性能测量和在线服务状态。

| 中文术语 | 英文或缩写 | 定义 | 首次学习 | 相关公式 |
| --- | --- | --- | --- | --- |
| 量化 | Quantization | 用低比特表示权重或激活以减少容量和带宽 | Chapter05 | [F-QUANT-01](./formulas.md#f-quant-01) |
| 缩放因子 | scale | 将量化整数映射回近似浮点范围的比例参数 | [Chapter05 Quantization Basics](../chapter05_quantization/01_quantization_basics.md) | [F-QUANT-01](./formulas.md#f-quant-01) |
| 反量化 | dequantization | 根据 scale、zero point 或 block metadata 恢复近似浮点值 | Chapter05 | [F-QUANT-01](./formulas.md#f-quant-01) |
| 仅权重量化 | weight-only quantization | 权重使用低比特格式、激活与输出仍保持浮点表示的推理方式 | [Chapter05 Quantized Linear](../chapter05_quantization/05_linear_with_quant.md) | [F-QUANT-02](./formulas.md#f-quant-02) |
| 混合量化 | mixed quantization | 根据不同 Tensor 的精度敏感性和执行支持选择不同位宽或格式 | [Chapter05 Quantization Basics](../chapter05_quantization/01_quantization_basics.md) | [F-QUANT-02](./formulas.md#f-quant-02) |
| K-quant | ggml K-quant | 使用 super-block、分组 scale 和 packed quant 的 ggml 格式族；名称与 K/V Cache 无关 | [Chapter05 Dequantization](../chapter05_quantization/04_dequantize_op.md) | [F-QUANT-01](./formulas.md#f-quant-01) |
| 激活量化 | activation quantization | 对 Prefill/Decode 产生的中间 Tensor 进行静态或动态低精度表示 | [Chapter05 Runtime Quantization](../chapter05_quantization/08_quantization_in_serving_runtime.md) | [F-QUANT-03](./formulas.md#f-quant-03) |
| K/V Cache 量化 | KV cache quantization | 用低精度 payload 和 scale metadata 保存请求级 key/value 历史 | [Chapter07 Production Extensions](../chapter07_kv_cache/next/06_production_extensions.md) | [F-QUANT-04](./formulas.md#f-quant-04) |
| 融合内核 | fused kernel | 在一次 kernel 或紧密循环中完成原本分离的多个阶段 | Chapter05、GPU 路线 | [F-QUANT-02](./formulas.md#f-quant-02) |
| 性能基准 | benchmark | 在固定 workload 和环境下测量延迟、吞吐与资源使用 | Chapter09 | [F-PERF-02](./formulas.md#f-perf-02) |
| 性能剖析 | profiling | 将性能结果归因到阶段、算子、带宽或同步 | Chapter09 | [F-PERF-02](./formulas.md#f-perf-02) |
| 首词元延迟 | Time to First Token, TTFT | 请求接收到首个生成词元的时间 | Chapter09-10 | [F-PERF-01](./formulas.md#f-perf-01) |
| 每词元输出时间 | Time per Output Token, TPOT | 首词元之后每个输出词元的平均耗时 | Chapter09-10 | [F-PERF-01](./formulas.md#f-perf-01) |
| 吞吐 | throughput | 单位时间完成的 token 数或请求数 | Chapter09 | [F-PERF-02](./formulas.md#f-perf-02) |
| 排队时间 | queue wait | 请求从 accepted 到获得执行机会的时间 | Chapter10 | [F-PERF-01](./formulas.md#f-perf-01) |
| 回压 | backpressure | 服务过载时限制新请求或输出速度的机制 | Chapter10 | - |
| 服务核心 | Service Core | 与网络 transport 解耦的 admission、调度、执行和指标核心 | Chapter10 | - |
| 传输层 | transport | HTTP/SSE 等连接、协议解析和字节发送实现 | Chapter10 | - |
| 准入控制 | admission control | 在请求进入等待队列前按容量或策略决定接受或拒绝 | Chapter10 | [F-SERVE-01](./formulas.md#f-serve-01) |
| 服务器发送事件 | Server-Sent Events, SSE | 服务端通过一个 HTTP 响应持续发送文本事件的单向流协议 | Chapter10 | - |
| 在途请求 | in-flight request | 已接受但尚未进入完成、失败、取消或超时终态的请求 | Chapter10 | [F-SERVE-01](./formulas.md#f-serve-01) |

## 多模态

这些术语描述图片如何转换为语言模型可以消费的 embedding。

| 中文术语 | 英文或缩写 | 定义 | 首次学习 | 相关公式 |
| --- | --- | --- | --- | --- |
| 多模态推理 | multimodal inference | 同一请求联合处理文本、图片或其他模态 | [多模态学习计划](../guides/multimodal_learning_plan.md) | [F-MM-01](./formulas.md#f-mm-01) |
| 图像块 | image patch | Vision Transformer 输入的固定大小局部像素块 | 多模态输入链路 | [F-MM-02](./formulas.md#f-mm-02) |
| 全局概览图 | overview image | 动态切图时保留整图关系的固定大小输入 | 多模态输入链路 | [F-MM-02](./formulas.md#f-mm-02) |
| 局部切片 | local tile | 从 refined image 裁出的固定大小高分辨率输入 | 多模态输入链路 | [F-MM-02](./formulas.md#f-mm-02) |
| 视觉编码器 | Vision Encoder | 将图像 patch 转换为视觉 hidden state 的 Transformer | 多模态输入链路 | [F-ATTN-02](./formulas.md#f-attn-02) |
| 视觉投影器 | Projector | 将视觉 hidden state 合并并映射到 LLM embedding width | 多模态输入链路 | [F-MM-02](./formulas.md#f-mm-02) |
| 视觉嵌入 | visual embedding | Projector 输出并插入语言模型序列的连续向量 | 多模态输入链路 | - |
| 图像 token 预算 | image-token budget | 控制图片像素预算、tile 数和上下文占用的运行参数 | 多模态输入链路 | [F-MM-01](./formulas.md#f-mm-01) |
| 多模态投影文件 | multimodal projector, mmproj | 保存 Vision Encoder 和 Projector metadata/权重的模型文件 | 多模态输入链路 | - |

## 验证与证据

这些术语区分教学实现本身与用于判断实现是否正确、是否更快的证据。

| 中文术语 | 英文或缩写 | 定义 | 首次学习 | 相关公式 |
| --- | --- | --- | --- | --- |
| 测试夹具 | fixture | 为测试固定的输入、配置和期望属性，可来自合成数据或外部工具 | Foundation、Chapter01 | - |
| 基准结果 | golden result | 由可信来源或已审核基线产生、用于数值或输出序列比较的期望结果 | Chapter01、Chapter09 | - |
| 参考实现 | reference implementation | 优先保证语义清晰和正确性、用于校验优化实现的实现路径 | Chapter03、Chapter05 | - |
| 回归测试 | regression test | 防止已通过行为在后续修改中退化的自动化检查 | Chapter09 | - |
| A/B 实验 | A/B experiment | 固定 workload 和环境，只改变一个目标因素的对照测量 | Chapter09 | [F-PERF-02](./formulas.md#f-perf-02) |

## 参考框架与格式

这些名称在教程中作为外部基线或实现策略对照，不代表教程依赖其全部组件。

| 名称 | 类型 | 在教程中的作用 |
| --- | --- | --- |
| GGUF | 模型文件格式 | 保存 metadata、tokenizer、tensor index 和量化 payload |
| llama.cpp | C/C++ 推理框架 | 本地 CPU/GPU、GGUF、Tokenizer 和多模态外部基线 |
| vLLM | Python 前端与 GPU engine | continuous batching、PagedAttention 和 serving 对照 |
| SGLang | 推理服务框架 | scheduler、prefix reuse 和应用编排对照 |
| TensorRT-LLM | NVIDIA 推理框架 | GPU engine、量化 kernel 和部署优化对照 |
| Ollama | 本地模型应用工具 | 模型打包、运行和交付体验对照 |

## 使用规则

1. 正文第一次出现术语时使用“中文名称（英文或缩写）”，后续优先使用中文名称。
2. C++ 类型、函数、字段和程序输出保留源码拼写并使用代码格式，例如 `TensorViewRef`、`Runtime::step()`；不要把代码名当作普通中文词语使用。
3. GGUF、CUDA、RMSNorm、RoPE、SwiGLU、llama.cpp 等格式、算法和项目专名保留标准拼写。
4. 图表以中文概念作为主标签；需要连接源码时，在次级标签中补充英文术语或代码名。
5. 性能术语必须同时说明时间区间、单位、请求数和词元口径。
6. 项目版本和实现状态从生态项目卡查询，术语表只解释稳定含义。
