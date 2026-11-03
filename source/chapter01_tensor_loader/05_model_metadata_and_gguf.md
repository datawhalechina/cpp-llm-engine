# 01.5 模型元数据与 GGUF

真实模型文件同时保存架构配置、tokenizer metadata、tensor index 和权重 payload。本节沿 GGUF Reader、Metadata Adapter、Tensor Loader 和 Model Builder 追踪这些信息如何变成后续章节使用的内部对象。

## GGUF 加载链

<a href="../_code/chapter01_tensor_loader/src/gguf_reader.cpp"><code>gguf_reader.cpp</code></a> 读取文件结构，<a href="../_code/chapter01_tensor_loader/src/model_metadata_adapter.cpp"><code>model_metadata_adapter.cpp</code></a> 归一化配置，<a href="../_code/chapter01_tensor_loader/src/model_builder.cpp"><code>model_builder.cpp</code></a> 检查配置与权重是否匹配。

![Chapter 01 GGUF 四层加载链](../assets/diagrams/chapter01-gguf-loading-chain.svg)

| 层 | 主要输入 | 主要输出 | 检查内容 |
| --- | --- | --- | --- |
| GGUF Reader | 文件字节 | header、KV、tensor index、offset | 二进制结构与数据区位置 |
| Metadata Adapter | GGUF key/value | `ModelMetadata` | 架构字段、层数、维度和特殊 token |
| 张量加载器 | 张量索引和数据区 | 张量映射 | 名称、形状、数据类型、偏移和权重数据 |
| 模型构建器 | 模型元数据、张量映射、分词器 | `LoadedModel`、`DecoderInput` | 词表、模型结构和 Llama 权重映射 |

Tokenizer metadata 来自同一份 GGUF，但使用独立接口读取 vocabulary、scores、token types、BOS/EOS 和可选 chat template。Tokenizer 规则不会混入权重读取过程。

外部格式经过 adapter 后统一为内部配置：

```text
GGUF Reader -> GGUF Metadata Adapter -> ModelMetadata
其他 Reader -> 对应 Metadata Adapter -> ModelMetadata
                                      -> ModelConfig
```

替换文件格式时，解码器和生成运行时继续消费 `ModelConfig` 和张量映射，不读取 GGUF 键或文件偏移。

## 配置、权重与模型布局

内部配置保存能够检查模型结构的稳定字段：

| 字段 | 用途 |
| --- | --- |
| `architecture` | 选择对应模型 schema |
| `vocab_size` | 校验 tokenizer、embedding 和输出层 |
| `hidden_size` | 校验 Tensor shape 和 Decoder hidden state |
| `max_position` | 限制输入长度和位置编号 |
| `layer_count` | 校验解码器块数量 |
| `bos_token_id` / `eos_token_id` | 固定输入输出的特殊 token 语义 |

TinyLlama 1.1B 提供本教程的第一条真实模型基线。`block_count=22` 对应 `blk.0` 到 `blk.21`；每个 Block 的九个权重 Tensor 由 Llama schema 检查，embedding、Final Norm 和输出权重位于 Block 堆栈之外。

文件解析和模型布局通过两条路径汇合：

```text
metadata adapter -> ModelMetadata -> ModelConfig
张量加载器       -> 张量映射      -> Llama 张量布局
                                  -> LoadedModel / DecoderInput
```

`DecoderInput` 保存稳定的 Tensor 名称映射，payload 仍由 `LoadedModel.tensors` 持有。Chapter04 根据这些配置和权重映射解释真实 Decoder 的计算；Chapter05 负责解码 Q4_K、Q6_K 等量化 payload。

## 验证与衔接

默认回归使用 fixture 检查 GGUF header、metadata、tensor index、alignment 和 Llama schema：

```bash
bash code/chapter01_tensor_loader/run.sh
```

已下载 TinyLlama GGUF 时，可以直接检查真实模型索引：

```bash
cmake -S code/chapter01_tensor_loader \
  -B /tmp/cpp-llm-engine-chapter01-build \
  -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/cpp-llm-engine-chapter01-build \
  --target gguf_inspect -j2
/tmp/cpp-llm-engine-chapter01-build/gguf_inspect \
  llama.cpp/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf \
  --validate-schema
```

验证结果应覆盖：metadata 能转换为 `ModelConfig`，Tensor Loader 能按索引读取支持的 payload，Llama schema 能发现缺失或 shape 错误的权重，Model Builder 能生成持有稳定映射的 `LoadedModel`。真实配置与权重由 [`Chapter04`](../chapter04_attention/06_real_tinyllama_decoder.md) 的 `code/model` 消费；[`Chapter02`](../chapter02_tensor_ops/intro.md) 使用独立 CPU Float32 教学 Tensor 学习 shape、Storage 和 View 契约，不承接 GGUF payload。
