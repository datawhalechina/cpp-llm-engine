# 真实模型执行模块

本目录连接 Chapter01 产生的模型配置、权重和 Tokenizer 资产，以及 Chapter05 提供的量化 Linear 后端。Chapter04 使用这里的真实稠密 Decoder 解释 RMSNorm、RoPE、GQA、SwiGLU、分层 K/V Cache 和 LM Head。

| 资产 | 职责 |
| --- | --- |
| `RealLlamaCpuModel` | 持有共享只读权重和模型结构规则 |
| `RealLlamaCpuContext` | 保存单个请求的 position、分层 K/V 和执行统计 |
| `RealLlamaCudaModel` | 按层上传并共享 CUDA 权重，不保存请求状态 |
| `RealLlamaCudaContext` | 独占 stream、层中间张量和 position，持有连续 K/V 或绑定分页数据面 |
| `gguf_first_layer_forward` | 用真实量化权重执行首层单词元前向 |
| `gguf_first_layer_multitoken` | 用真实量化权重观察 RoPE、因果注意力和增量 K/V |
| `gguf_decoder_multilayer` | 执行多层 Decoder，并可选与 llama.cpp 比较 |

CPU 类型暂时保留在 `chapter01` 命名空间，以兼容已有引擎和多模态调用方；CUDA 类型使用 `cpp_llm_engine::model`。两条路径都遵守“模型共享权重、Context 隔离请求状态”的结构。

首层示例和 CPU 模型同时接受 Llama 与 Qwen3 布局。Qwen3 路径根据元数据使用 16 个 Q 头、8 个 K/V 头和 128 维头空间，在 RoPE 前对每个 Q/K 头独立执行 RMSNorm，并使用前半维与后半维配对的 NeoX RoPE。运行 `bash code/chapter04_attention/scripts/run_qwen3_first_layer.sh` 可观察首层数据流；运行 `bash code/chapter04_attention/scripts/run_qwen3_decoder.sh` 可验证 28 层 logits 和连续四步贪心解码。

CPU 类名暂时保留 `RealLlamaCpuModel` 以兼容引擎和多模态调用方，但加载时会读取 GGUF 的 `general.architecture`，只接受已经实现的 `llama` 与 `qwen3`。执行器工厂已复用该分派进入 Chapter06 单请求生成和 Chapter10 CPU 服务；Qwen3 CUDA 仍待接入。

CUDA 模型持有量化 embedding、分层 Decoder 权重、final norm 和 LM Head；Context 从 token ID 开始执行，并保存请求级连续 K/V 或绑定 Chapter07 分页数据面。真实测试已完成 TinyLlama 2 层与 22 层 hidden/K/V/logits 数值对照，Chapter06–10 继续验证真实 Tokenizer、Prefill、贪心解码、分页 batch、服务调度和请求隔离。

`RealLlamaCudaContext::set_attention_path()` 在请求 Context 上选择默认三阶段 GQA 或融合在线 GQA。共享 Model 不保存该策略，因此两个 Context 可以复用相同权重独立执行 A/B；默认值保持三阶段，分页 K/V 路径暂不使用该开关。真实多层测试用两个 token 步骤核对 hidden、logits 和 greedy token。

`forward_hidden_sequence()` 接收连续 `[T,D]` embedding，逐层使用 `CudaDecoderBatchLayer::prefill_sequence()` 批量执行 RMSNorm、Linear、RoPE 和因果 GQA，并一次向每层连续 K/V 追加 T 个位置。TinyLlama 22 层、2-token Tensor Prefill 与顺序执行的最大误差为 `3.05e-05`。

`prefill_tokens()` 继续补上批量 embedding、最后一行 RMSNorm 和 LM Head，并在 Context 内按 token 数复用 Prefill workspace。22 层最终 logits 与顺序执行最大误差为 `1.19e-05`；reset 后重复相同 shape，workspace 创建次数保持 1。连续 CUDA Executor 已使用该入口，绑定分页数据面的请求仍保留逐 token 路径。
