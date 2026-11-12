# 真实模型执行模块

本目录连接 Chapter01 产生的模型配置、权重和 Tokenizer 资产，以及 Chapter05 提供的量化 Linear 后端。Chapter04 使用这里的真实稠密 Decoder 解释 RMSNorm、RoPE、GQA、SwiGLU、分层 K/V Cache 和 LM Head。

## 代码架构

| 资产 | 职责 |
| --- | --- |
| `RealLlamaCpuModel` | 持有共享只读权重和模型结构规则 |
| `RealLlamaCpuContext` | 保存单个请求的 position、分层 K/V 和执行统计 |
| `RealLlamaCudaModel` | 按层上传并共享 CUDA 权重，不保存请求状态 |
| `RealLlamaCudaContext` | 独占 stream、层中间张量和 position，持有连续 K/V 或绑定分页数据面 |
| `gguf_first_layer_forward` | 用真实量化权重执行首层单词元前向 |
| `gguf_first_layer_multitoken` | 用真实量化权重观察 RoPE、因果注意力和增量 K/V |
| `gguf_decoder_multilayer` | 执行多层 Decoder，并可选与 llama.cpp 比较 |
| `PackedExpertView` | 把 `[cols, rows, experts]` 的真实 packed 权重映射为单个专家的二维只读窗口 |
| `run_real_moe_feed_forward` | 先做 Router/Top-K，再按需执行命中专家和共享专家 |
| `run_real_moe_feed_forward_batch` | 先路由整个 token 集合，再按专家分发、执行并合并回原顺序；可切换逐行基线与分组路径 |
| `qwen15_moe_ffn_inspect` | 用 Qwen1.5-MoE layer-0 检查 60 选 4 与真实量化 FFN |
| `qwen15_moe_ffn_compare` | 捕获 llama.cpp Attention、Top-K、逐层输出与 logits，对照单层、两层或 24 层前向 |
| `RealQwen2MoeCpuModel` | 共享 GGUF 索引、embedding、最终归一化和 LM Head，执行时逐层加载专家权重 |
| `RealQwen2MoeCpuContext` | 保存单个请求的位置和 24 层历史 K/V，支持 Prefill、Decode 和 reset |
| `run_qwen2_moe_layer_batch_with_context` | 保持各请求 Attention/K/V 独立，在同一层合并 MoE 路由与专家分发 |
| `RealDeepSeek2LayerContext` | 保存单层位置、归一化 latent K/V 与已旋转共享 Key，供稠密与 MoE 层共用 |
| `run_deepseek2_dense_token_layer` | 连接 DeepSeek2 首个稠密层的 RMSNorm、YaRN partial RoPE、压缩 MLA、SwiGLU 与双残差 |
| `run_real_deepseek2_moe_feed_forward` | 连接 DeepSeek2 的全路由 Softmax、64 选 6、量化专家和无门控共享专家 |
| `run_deepseek2_moe_token_layer_batch` | 保持各请求压缩 MLA 历史独立，再把同层 FFN 输入按专家分组执行 |
| `run_deepseek2_moe_token_layer` | 复用同一 MLA/Context 主体，连接后续 MoE 层和双残差 |
| `RealDeepSeek2CpuModel` | 共享 GGUF 索引、embedding、最终归一化和 LM Head，按层执行 1 个稠密层与 26 个 MoE 层 |
| `RealDeepSeek2CpuContext` | 保存单个请求的位置和 27 层压缩 MLA 历史，支持 Prefill、Decode 和 reset |
| `CudaDeepSeek2DenseLayerWeights` | 完整上传稠密首层的 Q2_K/Q3_K/IQ4_NL Linear 与 Float32 Norm 后替换输出对象，提供 MLA CUDA 只读权重视图 |
| `CudaDeepSeek2MoeWeights` | 在加载阶段切分并上传 64 个 packed experts 与共享专家，保留 Host Router 控制面 |
| `CudaDeepSeek2MoeFeedForward` | 执行 Top-6 命中专家和共享专家的 CUDA Gate/Up/Down，并在完整成功后写入输出 |
| `CudaDeepSeek2MoeBatchWorkspace` | 按专家打包多个请求的 FFN 输入，复用批量 Gate/Up/Down 与共享专家临时张量 |
| `CudaDeepSeek2MoeLayer` | 连接压缩 MLA、Host Top-6、CUDA 专家执行、双残差和请求级缓存回退 |
| `RealDeepSeek2CudaModel` | 常驻第 0 层稠密权重、26 层 MoE 权重、Embedding、最终归一化和 LM Head |
| `RealDeepSeek2CudaBatchWorkspace` | 在 27 层 Model 上复用各 MoE 层的跨请求批量工作区，并记录执行次数 |
| `RealDeepSeek2CudaContext` | 独占执行流、中间 Tensor 与 27 层压缩 MLA 历史，整组提交或回退 |
| `CudaDeepSeek2DenseLayer` | 在请求级压缩 MLA Cache 上执行 YaRN、稠密 FFN 与双残差，并在失败时回退位置 |

## 所有权与执行边界

CPU 类型暂时保留在 `chapter01` 命名空间，以兼容已有引擎和多模态调用方；CUDA 类型使用 `cpp_llm_engine::model`。两条路径都遵守“模型共享权重、Context 隔离请求状态”的结构。

DeepSeek2 的稠密层和 MoE 层共享同一个按词元执行主体。Q 的 64 维位置分支和共享 Key 使用 GGUF 中的 YaRN factor、原始上下文长度与 log multiplier；归一化后的 512 维 latent K/V 和旋转后的共享 Key 只在整层成功后追加。第 0 层注入稠密 SwiGLU，后续 26 层注入全路由 Softmax、Top-6 和无门控共享专家；FFN 或残差失败时不提交 K/V、层输出或路由。Prefill 按层加载一次权重并顺序处理该层的全部词元，Context 只保存请求自己的压缩历史。`deepseek_v2_lite_layer_capture` 还会执行完整 27 层、最终 RMSNorm 和 LM Head；双词元 logits 与 llama.cpp 的余弦相似度为 `0.999616`，贪心词元同为 `185`。

CUDA 稠密首层保持相同状态边界。`CudaDeepSeek2DenseLayerWeights` 在模型加载阶段上传 packed Linear，并只为 K/V-B 建立一份共享 Float32 展开矩阵；`CudaDeepSeek2DenseLayer` 为请求保存中间 Tensor 和压缩历史。真实双词元 CPU/CUDA 层输出最大误差为 `5.01e-6`。

CUDA MoE 把 Router 和稳定 Top-K 留在 Host 控制面，将命中的 Top-6 路由专家与共享专家放到设备执行。模型加载阶段把三维 packed expert 切成独立设备矩阵，单请求工作区完成 Gate、Up、SwiGLU 和 Down；所有专家成功后才提交结果。真实第 1 层的专家索引与 CPU 一致，路由权重误差为 `0`，FFN 输出最大绝对误差为 `5.82e-10`。完整 MLA+MoE 层的双词元最大误差为 `9.83e-7`、平均误差为 `1.36e-7`，压缩缓存只在整层成功后推进。

跨请求路径在每层先分别完成 MLA，再由 `CudaDeepSeek2MoeBatchWorkspace` 按专家收集请求行。Gate、Up 和 Down 调用批量量化 Linear，共享专家一次处理完整批次，Host 最后按路由权重拆回请求输出。真实 batch=2 测试证明一个持久化工作区可连续服务 26 个 MoE 层，批量结果与相同输入的顺序路径一致。工作区还累计活跃专家组、多行专家组和路由行数；EXP-GPU-17 中两个相同请求的专家组全部形成双行批次。消除 MLA score 重算后，6 轮完整 Decode 配对中 grouped 路径比顺序路径改善约 `17.3%`。设备端 Top-K、加权拆回融合和单 kernel grouped-MoE 仍属于后续候选，但当前 profile 不支持把它们列为下一热点。

完整 CUDA Model 在共享对象中常驻 27 层只读权重，在每个请求 Context 中创建 27 份压缩 MLA Cache。单词元依次经过第 0 层稠密 FFN、后续 26 层 MoE、最终 RMSNorm 和 LM Head；任一层失败时，所有已推进层回退到调用前位置。真实 Q2_K 模型的 CPU/CUDA logits 余弦相似度为 `1`，贪心词元同为 `207`。Chapter06 Runtime、Chapter08 batch runner 和 Chapter10 ServiceCore 已分别验证生成状态、跨请求专家批执行和服务终态。

`RealDeepSeek2CpuModel` 提供有界 LRU 层权重缓存，统计容量、常驻层、常驻字节、命中和未命中。默认容量为 0，执行后立即释放当层权重。完整 Decode 固定按 0–26 层顺序访问，容量小于 27 时层的复用距离大于缓存容量，下一轮通常仍全部未命中；容量 27 可以消除后续 GGUF 层读取，但会额外常驻接近整套 Decoder 权重。该缓存管理模型权重，不保存请求 K/V。

## 阅读顺序与验证

先从 Llama/Qwen 的稠密模型建立 `Model -> Context -> Decoder Layer` 关系，再阅读 Qwen1.5-MoE 的 packed expert View，最后进入 DeepSeek 的压缩 MLA 与跨请求专家分组。每一步都先运行较小的层级入口，再运行完整模型、Runtime 和服务入口。

| 阶段 | 主要入口 | 观察重点 |
| --- | --- | --- |
| 稠密 Decoder | `bash code/chapter04_attention/scripts/run.sh` | RMSNorm、Attention、FFN、残差和连续 K/V |
| Qwen3 CUDA | `bash code/engine/scripts/run_qwen3_cuda_decoder.sh` | Q/K Norm、NeoX RoPE、28 层 hidden/K/V/logits |
| Qwen1.5-MoE CPU | `bash code/chapter04_attention/scripts/run_qwen15_moe_decoder_compare.sh` | Router、Top-K、packed expert 和共享专家 |
| DeepSeek CPU | `bash code/chapter04_attention/scripts/run_deepseek_v2_lite_layer_capture.sh` | YaRN、压缩 MLA、稠密/MoE 混合层 |
| DeepSeek CUDA 层级 | `bash code/engine/scripts/run_deepseek_v2_lite_cuda_dense_layer.sh` | CPU/CUDA 首层数值与压缩缓存增长 |
| DeepSeek CUDA 全模型 | `bash code/engine/scripts/run_deepseek_v2_lite_cuda_model.sh` | 27 层 logits、位置提交与失败回退 |
| DeepSeek CUDA 批处理 | `bash code/engine/scripts/run_deepseek_v2_lite_cuda_batch.sh` | 请求级 MLA 隔离、专家分组和工作区复用 |

以上 CUDA 命令可用 `CUDA_HOME` 指定 Toolkit。真实模型路径由对应脚本的环境变量或默认教程路径提供，运行前先阅读脚本开头的参数说明。

## 扩展入口

首层示例和 CPU 模型同时接受 Llama 与 Qwen3 布局。Qwen3 路径根据元数据使用 16 个 Q 头、8 个 K/V 头和 128 维头空间，在 RoPE 前对每个 Q/K 头独立执行 RMSNorm，并使用前半维与后半维配对的 NeoX RoPE。运行 `bash code/chapter04_attention/scripts/run_qwen3_first_layer.sh` 可观察首层数据流；运行 `bash code/chapter04_attention/scripts/run_qwen3_decoder.sh` 可验证 28 层 logits 和连续四步贪心解码。

CPU 类名暂时保留 `RealLlamaCpuModel` 以兼容引擎和多模态调用方，但加载时会读取 GGUF 的 `general.architecture`，只接受已经实现的 `llama` 与 `qwen3`。执行器工厂已复用该分派进入 Chapter06 单请求生成和 Chapter10 CPU 服务。Qwen3 CUDA 首层已通过非对称投影、Q/K Norm、NeoX RoPE 与真实 K/V 验收；28 层 Decoder 使用同一模型对象继续验证。

CUDA 模型持有量化 embedding、分层 Decoder 权重、final norm 和 LM Head；Context 从 token ID 开始执行，并保存请求级连续 K/V 或绑定 Chapter07 分页数据面。真实测试已完成 TinyLlama 2 层与 22 层 hidden/K/V/logits 数值对照，Chapter06–10 继续验证真实 Tokenizer、Prefill、贪心解码、分页 batch、服务调度和请求隔离。

`RealLlamaCudaContext::set_attention_path()` 在请求 Context 上选择默认三阶段 GQA 或融合在线 GQA。共享 Model 不保存该策略，因此两个 Context 可以复用相同权重独立执行 A/B；默认值保持三阶段，分页 K/V 路径暂不使用该开关。真实多层测试用两个 token 步骤核对 hidden、logits 和 greedy token。

`forward_hidden_sequence()` 接收连续 `[T,D]` embedding，逐层使用 `CudaDecoderBatchLayer::prefill_sequence()` 批量执行 RMSNorm、Linear、RoPE 和因果 GQA，并一次向每层连续 K/V 追加 T 个位置。TinyLlama 22 层、2-token Tensor Prefill 与顺序执行的最大误差为 `3.05e-05`。

`prefill_tokens()` 继续补上批量 embedding、最后一行 RMSNorm 和 LM Head，并在 Context 内按 token 数复用 Prefill workspace。22 层最终 logits 与顺序执行最大误差为 `1.19e-05`；reset 后重复相同 shape，workspace 创建次数保持 1。连续 CUDA Executor 已使用该入口，绑定分页数据面的请求仍保留逐 token 路径。

Qwen1.5-MoE 路径把 Chapter01 的 Router、三组 packed experts 和共享专家权重交给 `run_real_moe_feed_forward()`。执行器先对 60 个 Router 分数做 Softmax，再只为 Top-4 专家创建 View；Q3_K/Q4_K/Q5_0/Q5_1/Q5_K/Q6_K/Q8_0 均进入 Chapter05 统一 Linear。`run_qwen2_moe_prefill_layer()` 连接 RMSNorm、带 bias 的 Q/K/V、NeoX RoPE、因果 Attention、MoE 和双残差，并按词元返回旋转后的 K/V 负载。

多词元 MoE 先构造按专家索引的 assignment 表，每个活跃专家只创建一次 packed View，再把组内输入交给 Gate、Up、Down 的 batch Linear，最后将加权输出写回原 token 行。这里的“组”是同一层中命中同一专家的 `(token, route_slot)` 集合，不是请求组或 K/V 槽位；一个 Top-K 词元会进入 K 个专家组。Q3_K 与 Q5_K batch backend 会让同一个量化块只解码一次并服务多个输入行；其他格式按各自 backend 的 batch 能力执行。`RealMoeBatchMode` 在同一份权重和路由上切换 `RowWise` 与 `Grouped`，默认使用后者。

EXP-CPU-13～18 从第一层 MoE 子层逐步扩展到完整 Decoder、batch 1/2/4、多步 Decode、Chapter08 固定队列和 Chapter10 动态准入。各层数值与生成词元闸门均通过；性能结论限于归档 workload，不外推为开放环或 HTTP 吞吐。

多请求 Decode 采用按层交错：每个请求先从自己的历史 K/V 计算 Attention，再把该层所有请求的归一化 FFN 输入交给同一张专家 assignment 表，最后按请求拆回输出。全部请求完成后才提交本层 K/V；后续层或 LM Head 失败时，Model 恢复整批请求的 24 层状态。层级测试将该路径与逐请求执行逐元素对照。

DeepSeek-V2-Lite 的 CPU 与 CUDA 路径使用相同分组原则：Attention 保存各请求独立的压缩 latent K/V 和位置 Key，每层先逐请求完成 MLA，再把 64 选 6 的 FFN 输入交给共同的专家分发表。CPU 路径复用量化专家 View 和 batch Linear；CUDA 路径把同一专家的请求行交给设备端批量量化 Linear，并让共享专家处理完整批次。两者都保留 Host 路由与加权拆回，因此不等同于成熟框架的融合 grouped-MoE kernel。

24 层对照按层加载权重，避免 60 个专家的全部层权重同时常驻内存。局部路径使用 llama.cpp 上一层输出，24 层 Top-K 专家集合全部一致，最大局部 MAE 为 `0.0418066`、最低余弦相似度为 `0.999857`。累计路径保留教程输出，量化误差在 3 个词元位置触发路由分叉，最终 hidden MAE 为 `0.136931`、余弦相似度为 `0.997824`。最终 RMSNorm 与 Q6_K LM Head 已接入统一 Linear；累计 logits 余弦相似度为 `0.999401`，Top-5 前四项与贪心词元 `151643` 匹配 llama.cpp。

请求 Context 为每层分别保存旋转后的历史 K/V 和当前位置。整段 Prefill 与逐词元调用使用同一位置规则；任一层、最终归一化或 LM Head 失败时恢复全部 24 层状态。`RealQwen2MoeExecutorFactory` 已将该 Context 接入 Chapter06，真实 Tokenizer 文本闭环生成 `11 358` 并解码为 `, I`。默认路径按层从 GGUF 重载权重以控制常驻内存，当前不作服务并发性能承诺。

Model 提供显式 LRU 层权重缓存和命中、未命中、常驻字节统计，默认容量为 0。24 层顺序执行中，容量小于 24 通常会在下一轮到来前淘汰旧层，因此小缓存用于观察策略而不是默认优化；全层缓存可以跨 Decode 复用权重，但会让大部分 GGUF payload 常驻 CPU 内存。是否启用必须由调用方根据内存预算和 Chapter09 实验决定。
