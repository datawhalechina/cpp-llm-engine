# Chapter 01：Tensor Loader 模型权重与 Tensor 对象

本目录把 GGUF、metadata、tokenizer metadata 和权重索引转换为教程自己的稳定对象。完整学习说明见 [`source/chapter01_tensor_loader`](../../source/chapter01_tensor_loader/intro.md)。

## 本章资产总览

| 资产 | 作用 |
| --- | --- |
| `src/tensor.*` | Tensor shape、dtype、storage 和 payload |
| `src/model_loader.*` | 教学自定义格式的 Tensor loader |
| `src/gguf_reader.*` | GGUF header、KV 和 Tensor index reader |
| `src/gguf_tensor_loader.*` | F32/F16/I8/Q8_0/Q4_K/Q6_K 原始 payload loader |
| `src/model_metadata*` | 外部 metadata 到 `ModelMetadata` / `ModelConfig` |
| `src/tokenizer*` | tokenizer contract 和 metadata 边界 |
| `src/model_schema.*` | Llama Tensor layout 和 shape 校验 |
| `src/model_builder.*` | `LoadedModel` 和 `DecoderInput` 装配 |
| `examples/gguf_inspect.cpp` | 真实 GGUF 检查工具 |
| `src/real_layer_weights.*` | 按需装配真实 Llama decoder layer 权重 |
| `examples/gguf_first_layer_inspect.cpp` | 第一层九个 tensor 的 dtype/shape/payload 检查 |
| `src/real_tokenizer.*` | 可选 llama.cpp reference tokenizer adapter |
| `examples/llama_tokenizer_compare.cpp` | 真实 TinyLlama encode/decode 对照工具 |
| `examples/gguf_q6k_compare.cpp` | 真实 Q6_K tensor 与 llama.cpp 官方解码对照工具（可选整合构建） |
| `examples/gguf_first_layer_forward.cpp` | 真实 TinyLlama layer-0 单 token CPU forward（可选整合构建） |
| `examples/gguf_first_layer_multitoken.cpp` | 真实 TinyLlama layer-0 多 token CPU forward：RoPE、causal score、GQA、KV Cache（可选整合构建） |
| `../model` | 消费本章模型资产的真实 Decoder 执行模块 |
| `run_real_tokenizer.sh` | 构建并验证 TinyLlama 真实 tokenizer contract |
| `tests/chapter01_tensor_loader_test.cpp` | fixture、loader、schema 和装配测试 |
| `run.sh` | 隔离构建和 CTest 入口 |

## 运行与验证

从仓库根目录执行：

```bash
bash code/chapter01_tensor_loader/run.sh
```

检查真实 TinyLlama：

```bash
export CHAPTER01_BUILD_DIR=/tmp/cpp-llm-engine-chapter01-build
bash code/chapter01_tensor_loader/run.sh
"$CHAPTER01_BUILD_DIR/gguf_inspect" \
  llama.cpp/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf --validate-schema
```

该检查验证 metadata、201 个 Tensor index、22 层 Llama layout 和 `DecoderInput` mapping；不执行完整 Transformer Decoder。

检查真实第一层权重：

```bash
export CHAPTER01_BUILD_DIR=/tmp/cpp-llm-engine-chapter01-build
bash code/chapter01_tensor_loader/run.sh
"$CHAPTER01_BUILD_DIR/gguf_first_layer_inspect" \
  llama.cpp/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
```

该工具只加载 `blk.0` 的九个 tensor，不执行 Attention/FFN forward；它先固定真实权重进入 CPU 计算层前的 shape、dtype 和 payload 边界。

## 真实 tokenizer 对照

toy tokenizer 仍只用于接口测试。启用 llama.cpp reference adapter 后，使用真实 GGUF vocabulary：

```bash
cmake -S code/chapter01_tensor_loader \
  -B /tmp/cpp-llm-engine-chapter01-real-tokenizer \
  -DCHAPTER01_ENABLE_LLAMA_CPP=ON \
  -DLLAMA_CPP_ROOT="$PWD/llama.cpp"
cmake --build /tmp/cpp-llm-engine-chapter01-real-tokenizer
/tmp/cpp-llm-engine-chapter01-real-tokenizer/llama_tokenizer_compare \
  llama.cpp/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf "Hello TinyLlama"
```

Q6_K 真实 tensor 对照需要同时启用 Chapter 05 和 llama.cpp：

```bash
cmake -S code/chapter01_tensor_loader -B /tmp/chapter01-q6k \
  -DCHAPTER05_SOURCE_DIR="$PWD/code/chapter05_quantization" \
  -DCHAPTER01_ENABLE_LLAMA_CPP=ON \
  -DLLAMA_CPP_ROOT="$PWD/llama.cpp"
cmake --build /tmp/chapter01-q6k --target gguf_q6k_compare
/tmp/chapter01-q6k/gguf_q6k_compare \
  llama.cpp/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
```

真实首层单 token smoke 复用同一个 Chapter 05 整合构建：

```bash
cmake --build /tmp/chapter01-q6k --target gguf_first_layer_forward
/tmp/chapter01-q6k/gguf_first_layer_forward \
  llama.cpp/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
```

它会读取 token embedding、layer-0 的 Q4_K/Q6_K/Q8_0 权重，执行 RMSNorm、GQA、SwiGLU 和残差，并检查输出是否 finite。token-prefill 与预计算 embedding-prefill 汇合到同一个 Decoder hidden-state 路径；单 token smoke 仍不代表完整模型输出对照。

真实首层多 token smoke：

```bash
cmake --build /tmp/chapter01-q6k --target gguf_first_layer_multitoken
/tmp/chapter01-q6k/gguf_first_layer_multitoken \
  llama.cpp/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
```

该工具使用固定的 6 个 TinyLlama token，逐位置执行 RoPE、causal softmax 和 GQA，并把 K/V 追加到首层 KV Cache。输出中的 `attention_score_count`、`kv_cache_tokens` 和 `last_output_finite=true` 是本步的最小验证证据；它仍只执行 layer-0，不代表完整 Decoder 或 llama.cpp 数值等价。

Chapter04 的多层 Decoder smoke 默认执行前 2 层，传入 `22` 可验证完整 Block 堆栈和 logits：

```bash
cmake --build /tmp/chapter01-q6k --target gguf_decoder_multilayer
/tmp/chapter01-q6k/gguf_decoder_multilayer \
  llama.cpp/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf 22
```

第二个参数可以改为不超过模型层数的值。示例不再复制 RoPE、GQA、FFN 和 LM Head，而是加载一个 `RealLlamaCpuModel`，再由它创建独立的 `RealLlamaCpuContext` 执行 `prefill()` / `decode()`。Model 共享量化权重，Context 独占 position、分层 K/V 和统计；两个 Context 用同一 prompt 得到相同 logits，并输出 `context_isolation: PASS`。

Chapter01 到这里结束：它输出模型配置、Tensor Map、Tokenizer 和按层权重，不执行完整 Decoder。真实 CPU Decoder 位于 `code/model`，由 Chapter04 解释其数学结构和执行顺序。`RealLlamaCpuModel::decoder_contract()` 输出结构契约，`RealLlamaCpuContext` 实现 Core 的 `DecoderExecutionContext`，Engine 只依赖该执行接口，不读取 GGUF 或具体层类型。

当构建同时启用 `CHAPTER01_ENABLE_LLAMA_CPP=ON` 时，完整 22 层运行还会用相同 token IDs 调用 llama.cpp CPU reference。对照关闭 Flash Attention，并使用 F32 K/V Cache；验收条件覆盖 cosine similarity、mean absolute error、RMSE 和 top-5 overlap。当前 TinyLlama 结果为 cosine `0.99952`、RMSE `0.148398`、top-1 一致、top-5 `5/5`。随后两边按同一 greedy 规则生成 4 个 token，结果均为 `29892 306 29915 29885`，最终输出 `greedy_sequence_match: true`。不同 CPU kernel 的累加顺序会让单个 logit 存在误差，因此不把 `max_abs` 作为唯一判据。

也可以直接运行可重复检查：

`LLAMA_CPP_BUILD_DIR` 默认指向 `llama.cpp/build`；源码与构建产物分离时，可以把它设为 `/tmp` 下的实际构建目录。

```bash
bash code/chapter01_tensor_loader/run_real_tokenizer.sh
```

这一步调用 llama.cpp 的 tokenizer API 固定真实 token ids 和 detokenized 文本；它是 Chapter01 的 reference adapter，不复制 tokenizer 算法。后续自研 Runtime 只依赖 `TokenSequence`，可以在同一契约下替换实现。
