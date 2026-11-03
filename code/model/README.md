# 真实模型执行模块

本目录连接 Chapter01 产生的模型配置、权重和 Tokenizer 资产，以及 Chapter05 提供的量化 Linear 后端。Chapter04 使用这里的真实 Llama Decoder 解释 RMSNorm、RoPE、GQA、SwiGLU、分层 KV Cache 和 LM Head。

| 资产 | 职责 |
| --- | --- |
| `RealLlamaCpuModel` | 持有共享只读权重和模型结构契约 |
| `RealLlamaCpuContext` | 保存单个请求的 position、分层 K/V 和执行统计 |
| `gguf_decoder_multilayer` | 执行多层 Decoder，并可选与 llama.cpp 比较 |

类型暂时保留在 `chapter01` 命名空间，以兼容已有 Engine 和多模态调用方；模块归属和构建依赖已经与 Chapter01 Loader 分离。
