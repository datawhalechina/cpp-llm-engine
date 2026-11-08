# 模型基线与扩展顺序

本教程用模型来验证推理引擎对象，不比较模型回答质量。默认模型必须能在普通 CPU 环境复现完整链路；其他模型只有在 Loader、Tokenizer、Decoder 和测试证据都接通后，才标记为教程实现支持。

## 当前模型状态

| 模型 | 架构 | 在教程中的角色 | 当前状态 |
| --- | --- | --- | --- |
| TinyLlama 1.1B Chat Q4_K_M | Llama 稠密 Decoder | 默认 CPU 基线；固定 GGUF、Tokenizer、22 层 Decoder、Runtime 和服务结果 | 已接入默认主线 |
| Qwen3-0.6B | Qwen3 稠密 Decoder | 第二种小模型兼容目标；验证教程是否摆脱 Llama 专用假设 | 规划中，不能使用现有 Llama Decoder 直接运行 |
| DeepSeek-R1-Distill-Qwen-1.5B | Qwen2.5 稠密 Decoder | 在 Qwen 兼容路径上验证 tokenizer、聊天模板和模型配置变化 | 规划中；它是稠密蒸馏模型，不用于学习 MoE |
| Qwen3-30B-A3B | Qwen3 MoE | 对照 Router、Top-K 专家选择与专家执行 | 外部框架对照；30B 总权重不属于当前单卡运行承诺 |
| DeepSeek-V2-Lite | DeepSeek MoE + MLA | 对照 MoE 与 MLA 组合后的模型执行差异 | 进阶设计参照；16B 总权重且同时引入 MLA |

TinyLlama 继续作为默认模型，不是因为其他模型不能推理，而是因为当前的 Llama GGUF schema、权重名称、Tokenizer golden 和真实 Decoder 测试已经形成一条可重复证据链。直接替换模型文件会把格式差异、模型结构差异和数值错误混在一起。

## 接入新稠密模型需要改什么

Qwen3-0.6B 仍使用 RMSNorm、RoPE、分组查询注意力和 SwiGLU，但层数、隐藏维度、词表、特殊词元、聊天模板、RoPE 参数及权重命名与 TinyLlama 不同。兼容工作按数据流分散到已有章节，不新增一条平行教程：

| 章节 | 需要增加的兼容证据 |
| --- | --- |
| Chapter00 | 用 llama.cpp 建立固定版本、固定提示词的外部结果 |
| Chapter01 | 识别模型架构，解析对应 metadata、Tokenizer、聊天模板和权重 schema |
| Chapter04 | 由配置构造 Qwen Decoder，并验证层数、Attention 与 FFN 拓扑 |
| Chapter05 | 为选定 GGUF 量化类型建立 payload 和 Linear 证据 |
| Chapter06 | 核对预填充、逐词元解码、停止词元与采样结果 |
| Chapter09 | 保存版本化 golden、正确性比较和性能记录 |
| Chapter10 | 复用 ServiceCore；只新增模型选择与请求输入验证 |

DeepSeek-R1-Distill-Qwen-1.5B 应排在 Qwen 稠密模型之后。它基于 Qwen2.5 架构并调整了配置和 tokenizer，适合验证架构适配是否可复用，但不能证明教程已经支持 DeepSeek-V2/V3 的 MoE 执行。

## MoE 学习顺序

MoE 的教学对象不是模型名称，而是一次词元怎样经过 Router、选择 Top-K 专家、按专家重排、执行 FFN，再按路由权重合并。建议按以下顺序建设：

1. 在小尺寸固定输入上实现 Router、Top-K 和加权合并，并为并列分数、非法专家编号和空批次建立测试。
2. 增加专家分组与 dispatch/combine 索引，观察每个专家接收的词元数和临时内存。
3. 把教学 MoE FFN 作为 Chapter04 Decoder 的可选分支，不改变稠密主线。
4. 在 Chapter08 讨论多请求批次中的专家负载与调度，在 Chapter09记录吞吐、内存和负载分布。
5. 最后选择一个真实 MoE 格式做外部框架基线；只有总权重、量化格式和设备内存满足单机条件时，才进入自研真实模型执行。

“激活参数较少”只表示每个词元执行少量专家，不表示其余专家权重无需加载。Qwen3-30B-A3B 每个词元激活约 3B 参数，但仍有约 30B 总参数；DeepSeek-V2-Lite 也有 16B 总参数，并额外引入多头潜在注意力（MLA）。两者都不适合作为 TinyLlama 的直接替代品。

## 版本与资料

模型接入必须记录模型仓库、revision、文件校验值、量化类型、Tokenizer 配置和聊天模板，避免上游文件更新后 golden 无法复现。架构参数以模型发布方资料为准：

- [Qwen3 官方说明](https://qwenlm.github.io/blog/qwen3/)与 [Qwen3-0.6B 配置](https://huggingface.co/Qwen/Qwen3-0.6B/blob/main/config.json)
- [DeepSeek-R1-Distill-Qwen-1.5B 模型卡](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B)
- [DeepSeek-V2 官方仓库](https://github.com/deepseek-ai/DeepSeek-V2)
