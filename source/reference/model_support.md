# 模型基线与扩展顺序

本教程用模型验证推理引擎对象和执行链路。一个模型被标记为“教程支持”，需要同时接通加载器（Loader）、分词器（Tokenizer）、解码器（Decoder）和固定验证结果；默认模型还需要在普通 CPU 环境复现。

## 当前模型状态

| 模型 | 架构 | 在教程中的角色 | 当前状态 |
| --- | --- | --- | --- |
| TinyLlama 1.1B Chat Q4_K_M | Llama 稠密解码器 | 默认 CPU 基线；固定 GGUF、分词器、22 层解码器、运行时和服务结果 | 已接入默认主线 |
| Qwen3-0.6B | Qwen3 稠密解码器 | 第二种小模型兼容目标；暴露 Llama 专用假设 | 规划：新增 Qwen 配置、权重映射和解码器 |
| DeepSeek-R1-Distill-Qwen-1.5B | Qwen2.5 稠密解码器 | 验证 Qwen2.5 分词器、聊天模板和模型配置变化 | 规划：复用 Qwen 稠密模型适配路径 |
| Gemma | Gemma 稠密解码器 | 比较另一种模型家族的元数据、权重命名、归一化和 FFN 差异 | 外部兼容性参照 |
| Qwen3-30B-A3B | Qwen3 MoE | 对照路由器（Router）、Top-K 专家选择与专家执行 | 外部框架对照；总权重超出当前单卡目标 |
| DeepSeek-V2-Lite | DeepSeek MoE + MLA | 对照 MoE 与 MLA 组合后的模型执行差异 | 进阶设计参照；16B 总权重且同时引入 MLA |

TinyLlama 继续作为默认模型，因为当前 Llama GGUF 字段模式、权重名称、分词器固定结果和真实解码器测试已有可重复证据。接入另一模型时应逐项处理格式和结构差异，避免把兼容问题误判为数值错误。

## 接入新稠密模型需要改什么

Qwen3-0.6B 同样使用 RMSNorm、RoPE、分组查询注意力和 SwiGLU，但层数、隐藏维度、词表、特殊词元、聊天模板、RoPE 参数及权重命名与 TinyLlama 不同。兼容工作按数据流进入已有章节：

| 章节 | 需要增加的兼容证据 |
| --- | --- |
| Chapter00 | 用 llama.cpp 建立固定版本、固定提示词的外部结果 |
| Chapter01 | 识别模型架构，解析对应元数据、分词器、聊天模板和权重字段模式 |
| Chapter04 | 由配置构造 Qwen 解码器，并验证层数、注意力与 FFN 拓扑 |
| Chapter05 | 为选定 GGUF 量化类型建立数据布局和 Linear 证据 |
| Chapter06 | 核对预填充、逐词元解码、停止词元与采样结果 |
| Chapter09 | 保存版本化固定结果、正确性比较和性能记录 |
| Chapter10 | 复用 ServiceCore；只新增模型选择与请求输入验证 |

DeepSeek-R1-Distill-Qwen-1.5B 排在 Qwen 稠密模型之后。它基于 Qwen2.5 架构并调整配置和分词器，用于验证架构适配能否复用；DeepSeek-V2/V3 的 MoE 执行需要另一组路由与专家证据。

Gemma 也经过同一条接入链：Chapter01 识别架构、元数据和权重字段模式，Chapter04 根据配置构造对应解码器，Chapter05 核对量化格式。llama.cpp 或 gemma.cpp 的外部结果作为基线，本教程仍需独立完成上述适配与验证。

## MoE 学习顺序

MoE 教学聚焦一次词元怎样经过路由器、选择 Top-K 专家、按专家重排、执行 FFN，再按路由权重合并。建议按以下顺序建设：

1. 在小尺寸固定输入上实现 Router、Top-K 和加权合并，并为并列分数、非法专家编号和空批次建立测试。
2. 增加专家分组与分发/合并索引，观察每个专家接收的词元数和临时内存。
3. 把教学 MoE FFN 作为 Chapter04 解码器的可选分支，保持稠密主线的调用方式。
4. 在 Chapter08 讨论多请求批次中的专家负载与调度，在 Chapter09 记录吞吐、内存和负载分布。
5. 最后选择一个真实 MoE 格式做外部框架基线；只有总权重、量化格式和设备内存满足单机条件时，才进入自研真实模型执行。

“激活参数较少”表示每个词元只执行少量专家，模型部署仍需容纳其余专家权重。Qwen3-30B-A3B 每个词元激活约 3B 参数，总参数约 30B；DeepSeek-V2-Lite 总参数约 16B，并额外引入多头潜在注意力（MLA）。因此它们适合作为进阶设计参照，而非第一阶段单卡基线。

## 版本与资料

模型接入记录模型仓库、revision、文件校验值、量化类型、分词器配置和聊天模板，使上游文件更新后仍能复现固定结果。架构参数以模型发布方资料为准：

- [Qwen3 官方说明](https://qwenlm.github.io/blog/qwen3/)与 [Qwen3-0.6B 配置](https://huggingface.co/Qwen/Qwen3-0.6B/blob/main/config.json)
- [DeepSeek-R1-Distill-Qwen-1.5B 模型卡](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B)
- [DeepSeek-V2 官方仓库](https://github.com/deepseek-ai/DeepSeek-V2)
- [Gemma 模型文档](https://ai.google.dev/gemma/docs)与 [gemma.cpp](https://github.com/google/gemma.cpp)
