# model 真实模型执行代码导航

`code/model` 消费 Chapter01 的模型配置、权重和 Tokenizer 资产，使用 Chapter05 的量化 Linear，执行 Chapter04 讲解的真实 Llama Decoder。

## 阅读顺序

1. 阅读 `real_llama_cpu_decoder.h`，区分共享只读 Model 与请求独占 Context
2. 阅读 `real_llama_cpu_decoder.cpp`，跟踪 RMSNorm、RoPE、GQA、SwiGLU、分层 KV 和 LM Head
3. 阅读 `gguf_decoder_multilayer.cpp`，查看多层执行、Context 隔离和 llama.cpp 数值对照

## 核心代码

| 文件 | 作用 |
| --- | --- |
| <a href="/_code/model/include/cpp_llm_engine/model/real_llama_cpu_decoder.h"><code>real_llama_cpu_decoder.h</code></a> | 真实 Model、请求 Context、结构契约与 profiling 接口 |
| <a href="/_code/model/src/real_llama_cpu_decoder.cpp"><code>real_llama_cpu_decoder.cpp</code></a> | 22 层 CPU Decoder 与请求级 KV Cache |
| <a href="/_code/model/examples/gguf_decoder_multilayer.cpp"><code>gguf_decoder_multilayer.cpp</code></a> | 多层 logits、greedy token 与 llama.cpp 对照 |
| <a href="/_code/model/CMakeLists.txt"><code>CMakeLists.txt</code></a> | Loader、量化后端和真实模型执行目标 |

## 对应文档

- [Chapter01 模型权重与配置](../chapter01_tensor_loader/intro.md)
- [Chapter04 真实 TinyLlama Decoder](../chapter04_attention/06_real_tinyllama_decoder.md)
- [Chapter06 Generation Runtime](../chapter06_generation/intro.md)
