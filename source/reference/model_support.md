# 模型基线与扩展顺序

本教程用模型验证推理引擎对象和执行链路。一个模型被标记为“教程支持”，需要同时接通加载器（Loader）、分词器（Tokenizer）、解码器（Decoder）和固定验证结果；默认模型还需要在普通 CPU 环境复现。

## 当前模型状态

| 模型 | 架构 | 在教程中的角色 | 当前状态 |
| --- | --- | --- | --- |
| TinyLlama 1.1B Chat Q4_K_M | Llama 稠密解码器 | 默认 CPU 基线；固定 GGUF、分词器、22 层解码器、运行时和服务结果 | 已接入默认主线 |
| Qwen3-0.6B | Qwen3 稠密解码器 | 第二种小模型兼容目标；暴露 Llama 专用假设 | CPU 与 CUDA 推理服务纵向闭环已通过 |
| DeepSeek-R1-Distill-Qwen-1.5B | Qwen2.5 稠密解码器 | 验证 Qwen2.5 分词器、聊天模板和模型配置变化 | 规划：复用 Qwen 稠密模型适配路径 |
| Gemma | Gemma 稠密解码器 | 比较另一种模型家族的元数据、权重命名、归一化和 FFN 差异 | 外部兼容性参照 |
| Qwen1.5-MoE-A2.7B-Chat | Qwen2 MoE | 首个单卡 MoE 目标；学习 Router、Top-K、共享专家和专家执行 | CPU 24 层生成、按专家分组批执行、调度与服务闭环已接通 |
| Qwen3-30B-A3B | Qwen3 MoE | 对照路由器（Router）、Top-K 专家选择与专家执行 | 外部框架对照；总权重超出当前单卡目标 |
| DeepSeek-V2-Lite | DeepSeek MoE + MLA | 对照压缩 K/V、混合稠密/MoE 层与共享专家 | CPU 27 层 Decoder、生成、跨请求专家分组与 HTTP/SSE 服务已接通 |

TinyLlama 继续作为默认模型，因为当前 Llama GGUF 字段模式、权重名称、分词器固定结果和真实解码器测试已有可重复证据。接入另一模型时应逐项处理格式和结构差异，避免把兼容问题误判为数值错误。

## 接入新稠密模型需要改什么

Qwen3-0.6B 同样使用 RMSNorm、RoPE、分组查询注意力和 SwiGLU，但层数、隐藏维度、词表、特殊词元、聊天模板和 RoPE 参数与 TinyLlama 不同。它还为每层 Q/K 增加独立 RMSNorm，并允许 LM Head 与词嵌入共享权重。兼容工作按数据流进入已有章节：

当前固定 Qwen3 GGUF 包含 28 个解码器块、1024 维隐藏状态、16 个查询头、8 个 K/V 头和 128 维头空间。Chapter01 已验证 311 个张量、`Hello Qwen3` 的固定词元和聊天模板；Chapter04 已执行首层 Q/K Norm、NeoX RoPE、GQA、增量 K/V 和完整 28 层 CPU Decoder。默认四词元输入的 logits 与 llama.cpp 对照满足误差门槛，前五候选完全重合，连续四步贪心词元一致。

教程量化 Linear 保留 Float32 激活，llama.cpp 的 K-quant 点积会量化激活，因此两条路径不要求逐位相等。验收同时约束平均绝对误差、RMSE、余弦相似度、Top-5 重合和贪心序列；Chapter06 进一步验证真实 Tokenizer、Prefill、Decode、停止和请求隔离；Chapter10 验证 Completion、SSE、双请求调度、分页 K/V、Radix 命中和终态释放。Qwen3 的 CPU 与 CUDA 推理服务纵向闭环均已通过；CUDA 全层 logits 最大误差为 `4.95911e-05`，四个生成词元与 CPU golden 一致，服务组合验收耗时约 `0.80 s`。

| 章节 | 需要增加的兼容证据 |
| --- | --- |
| Chapter00 | 用 llama.cpp 建立固定版本、固定提示词的外部结果 |
| Chapter01 | 识别模型架构，解析对应元数据、分词器、聊天模板和权重字段模式 |
| Chapter04 | 由配置构造 Qwen 解码器，并验证层数、注意力与 FFN 拓扑 |
| Chapter05 | 为选定 GGUF 量化类型建立数据布局和 Linear 证据 |
| Chapter06 | 核对预填充、逐词元解码、停止词元与采样结果 |
| Chapter09 | 保存版本化固定结果、正确性比较和性能记录 |
| Chapter10 | 复用 ServiceCore；只新增模型选择与请求输入验证 |

DeepSeek-R1-Distill-Qwen-1.5B 排在 Qwen 稠密模型之后。它基于 Qwen2.5 架构并调整配置和分词器，用于验证架构适配能否复用。DeepSeek-V2-Lite 已完成外部基线、Loader、MLA reference、27 层混合 Decoder、LM Head、可选层权重缓存、Chapter06 单请求生成、Chapter08 双请求专家分组，以及 Chapter10 动态准入与 HTTP/SSE 回环；EXP-CPU-19 已补完整 Decode 配对证据，后续工作集中在 CUDA 数据面。

Gemma 也经过同一条接入链：Chapter01 识别架构、元数据和权重字段模式，Chapter04 根据配置构造对应解码器，Chapter05 核对量化格式。llama.cpp 或 gemma.cpp 的外部结果作为基线，本教程仍需独立完成上述适配与验证。

## MoE 学习顺序

MoE 教学聚焦一次词元怎样经过路由器、选择 Top-K 专家、按专家重排、执行 FFN，再按路由权重合并。建议按以下顺序建设：

首个真实目标选择 Qwen1.5-MoE-A2.7B-Chat。该模型总参数约 14.3B、每个词元激活约 2.7B；当前固定的 Q3_K_M GGUF 约 7.44GB，可以在本教程的 12GB 单卡目标上建立量化外部基线。Chapter01 已验证 24 层、411 个张量和 60 选 4 布局；Chapter04/05 已以双词元 24 层对照验证 NeoX RoPE、因果 Attention、分层 K/V、MoE、残差、最终 RMSNorm、Q6_K LM Head 和混合量化 Linear。Chapter06 已接入请求级 Context、真实 Tokenizer、Prefill、Decode 与贪心生成；Chapter08/10 继续验证跨请求专家分组、连续调度、动态准入和终态释放。

```bash
bash code/chapter00_quickstart/download_qwen15_moe.sh
bash code/chapter00_quickstart/run_qwen15_moe_baseline.sh
bash code/chapter04_attention/scripts/run_qwen15_moe_decoder_compare.sh
bash code/engine/scripts/run_qwen15_moe_generation.sh
```

1. 已用 llama.cpp 运行固定 Qwen1.5-MoE GGUF，并记录架构、文件校验值和生成结果。
2. Chapter01 已建立独立 MoE Schema，核对真实 Router、专家和共享专家权重。
3. Chapter04 已在小尺寸固定输入上实现稳定 Top-K、选中专家与共享专家加权合并，并覆盖并列分数、非法 Top-K、非有限输入和失败原子性。
4. Chapter05 已把首层真实量化专家权重接入统一 Linear，并用 llama.cpp 中间张量核对 Top-4 与共享专家合并结果。
5. Chapter04 已用双词元验证 NeoX RoPE、因果 Attention、连续 K/V 负载和完整 layer-0 输出。
6. Chapter04 已顺序加载 24 层权重；局部路径的 Top-K 全部匹配 llama.cpp，累计路径记录量化误差触发的路由分叉和最终隐藏状态误差。
7. 最终 RMSNorm 与 Q6_K LM Head 已通过 logits、Top-5 和贪心词元对照。
8. 每请求 Context 保存 24 层历史 K/V；Chapter06 Runtime 与真实 Tokenizer 文本闭环已经通过。
9. 按专家分组的分发/合并索引已接入 Chapter08 批执行；Chapter09 的 EXP-CPU-13～18 依次验证 MoE 子层、完整 Decoder、batch 范围、多步 Decode、固定队列和 Chapter10 动态准入。

当前分组路径是 CPU 默认实现，逐词元路径作为同权重、同路由的正确性和性能对照保留。实验覆盖最多四个请求和四步 Decode，并未建立开放环到达率或 HTTP 并发吞吐结论。

“激活参数较少”表示每个词元只执行少量专家，模型部署仍需容纳其余专家权重。Qwen3-30B-A3B 每个词元激活约 3B 参数，总参数约 30B。DeepSeek-V2-Lite 总参数约 16B，并额外引入多头潜在注意力（MLA）；固定 Q2_K GGUF 约 6.43GB，当前用于 CPU reference 和生成闭环，尚未形成 CUDA 单卡执行承诺。

## DeepSeek-V2-Lite 当前接入层级

```bash
bash code/chapter00_quickstart/download_deepseek_v2_lite.sh
bash code/chapter00_quickstart/run_deepseek_v2_lite_baseline.sh
bash code/chapter01_tensor_loader/run_deepseek_v2_lite_schema.sh
bash code/chapter04_attention/scripts/run_deepseek_v2_lite_layer_capture.sh
bash code/engine/scripts/run_deepseek_v2_lite_generation.sh
bash code/chapter08_scheduling/scripts/run_deepseek_v2_lite_batch.sh
bash code/chapter09_performance/scripts/run_deepseek2_moe_batch_runtime_ab.sh
bash code/chapter10_serving/scripts/run_deepseek_v2_lite_service.sh
bash code/engine/scripts/run_deepseek_v2_lite_http_server.sh --check
```

Chapter00 固定社区 GGUF revision、文件大小和 SHA256，并用 llama.cpp 建立短文本外部结果。Chapter01 识别 `deepseek2` 架构，归一化旧版与新版 MLA 元数据，检查 27 层、377 个张量、首层稠密 FFN、后续 64 选 6 专家和共享专家，并装配组合或拆分 K/V-B 权重。Chapter04 已用 CPU reference 证明低秩缓存与显式展开 Attention 数值一致。Chapter05 已实现固定 GGUF 所需的 Q2_K 与 IQ4_NL reference、scalar fused 和 policy。双词元第 0 层输出 MAE 为 `0.000891898`；第 1 层最终输出 MAE 为 `0.00128932`、余弦相似度为 `0.999859`。完整 27 层 logits 与 llama.cpp 的余弦相似度为 `0.999616`，贪心词元同为 `185`。Chapter06 使用真实 Tokenizer 和同一 Generation Runtime 从 `Hello` 生成固定词元 `11 304`，解码为 `, I`。Chapter08 验证两个独立 MLA Context 进入同一次 Decode batch，Chapter09 用 EXP-CPU-19 登记完整 Decode 配对结果，Chapter10 验证动态准入、Completion、SSE、指标和 socket 回环。

## 版本与资料

模型接入记录模型仓库、revision、文件校验值、量化类型、分词器配置和聊天模板，使上游文件更新后仍能复现固定结果。架构参数以模型发布方资料为准：

- [Qwen3 官方说明](https://qwenlm.github.io/blog/qwen3/)与 [Qwen3-0.6B 配置](https://huggingface.co/Qwen/Qwen3-0.6B/blob/main/config.json)
- [DeepSeek-R1-Distill-Qwen-1.5B 模型卡](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B)
- [DeepSeek-V2 官方仓库](https://github.com/deepseek-ai/DeepSeek-V2)
- [Gemma 模型文档](https://ai.google.dev/gemma/docs)与 [gemma.cpp](https://github.com/google/gemma.cpp)
