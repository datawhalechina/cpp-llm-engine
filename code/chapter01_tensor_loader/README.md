# Chapter 01：模型加载与输入对象

本目录把教学格式或 GGUF 中的配置、分词信息和权重转换为教程内部对象。完整学习顺序见 [`source/chapter01_tensor_loader`](../../source/chapter01_tensor_loader/intro.md)。

## 代码架构

教学链使用小型 CIE1 文件展示完整加载过程；真实链把 GGUF 结构解析、权重载荷读取和模型装配分开。两条路径都输出可检查的张量与模型配置。稠密模型由 `ModelBuilder` 装配整体布局，MoE 和 DeepSeek 模型由分层权重加载器交付对应层对象；解码器从 `code/model` 开始执行。

```text
CIE1 bytes -> ModelLoader -----------------------> Tensor map

GGUF bytes -> GgufReader -> Metadata/Tokenizer adapters -> ModelConfig
                         -> GgufTensorLoader -------------> Tensor map
                                                              |
                         +------------------------------------+
                         |                                    |
                  ModelBuilder                    real_layer_weights
                  稠密整体布局                    稠密 / MoE / MLA 分层对象
                         |                                    |
              LoadedModel + DecoderInput              Model 执行层
```

读取器解释外部格式，适配器把外部字段转换为内部配置，加载器检查权重载荷边界，构建器检查模型结构。任一步失败时，调用方都不会得到半加载对象。

`chapter01_tensor_loader` 是供 Model、多模态和 Engine 复用的库目标。直接构建本章时，`CHAPTER01_BUILD_STANDALONE_TARGETS` 默认开启并生成检查工具与测试；上层模块可以关闭该选项，只消费 Loader 库，避免重复构建本章入口。

### 设计分层

| 层次 | 代码对象 | 职责 |
| --- | --- | --- |
| 文件读取 | `GgufReader` | 解析文件头、键值和张量索引 |
| 字段适配 | metadata/tokenizer adapters | 把不同架构的 GGUF 键转换为内部配置 |
| 载荷读取 | `GgufTensorLoader` | 校验偏移与字节范围，保留浮点或量化权重载荷 |
| 结构检查 | `ModelSchema` | 检查模型家族要求的权重名称、形状和层次关系 |
| 模型装配 | `ModelBuilder`、`real_layer_weights` | 建立稠密整体映射或按需装配一层权重 |
| 文本转换 | `Tokenizer`、`RealTokenizer` | 在文本与词元编号之间转换，并保持词表约束 |

### 架构能力矩阵

| 模型家族 | 结构约束 | 权重交付方式 | 本章验证重点 |
| --- | --- | --- | --- |
| Llama / TinyLlama | 稠密 Decoder | 整体 `DecoderInput` 与按层稠密权重 | 22 层、GQA、词嵌入和输出权重 |
| Qwen3 | 稠密 Decoder | 整体布局与按层稠密权重 | Q/K Norm、投影宽度、共享输出权重 |
| Qwen2MoE / Qwen1.5-MoE | MoE Decoder | 按层 Router、打包专家和共享专家 | 60 选 4、专家形状和混合量化载荷 |
| DeepSeek2 / DeepSeek-V2-Lite | MLA + Dense/MoE 混合层 | 按层稠密或 MoE 权重 | 压缩 K/V、首层 Dense、后续 64 选 6 |

多模态 mmproj 复用 GGUF 读取能力；视觉结构约束和权重对象位于 `code/multimodal`，语言模型结构约束位于 `ModelSchema`。

## 阅读顺序与资产

先读 `tensor.*` 和 `model_loader.*`，用小文件理解对象与失败路径；再沿 GGUF Reader、Adapter、Tensor Loader、Model Builder 阅读真实加载链；最后查看检查工具和测试。

| 资产 | 作用 |
| --- | --- |
| `src/tensor.*` | 保存名称、形状、数据类型、存储和权重载荷 |
| `src/model_loader.*` | 加载 CIE1 教学格式并验证记录完整性 |
| `src/gguf_reader.*` | 读取 GGUF 文件头、模型元数据和张量索引 |
| `src/model_metadata*`、`src/tokenizer_metadata*` | 把 GGUF 字段转换为内部配置 |
| `src/gguf_tensor_loader.*` | 按索引读取 F32/F16/I8 与 Q2_K/Q3_K/Q4_K/Q5_0/Q5_1/Q5_K/Q6_K/Q8_0/IQ4_NL 权重载荷 |
| `src/model_schema.*` | 检查 Llama/Qwen3 稠密布局、Qwen2MoE 专家布局与 DeepSeek2 MLA 混合布局 |
| `src/real_layer_weights.*` | 按层装配稠密层、Qwen MoE 层和 DeepSeek2 稠密/MoE 层权重 |
| `src/model_builder.*` | 为 Llama/Qwen3 稠密模型装配 `LoadedModel` 和 `DecoderInput` |
| `examples/gguf_inspect.cpp` | 检查真实 GGUF 配置、索引和 schema |
| `examples/gguf_first_layer_inspect.cpp` | 检查首层九个权重的 dtype、shape 和 payload |
| `tests/chapter01_*_test.cpp` | 分别验证 Loader、Schema、Tokenizer/Payload 契约 |

## 运行与验证

从仓库根目录运行默认测试：

```bash
bash code/chapter01_tensor_loader/run.sh
```

脚本隔离构建并运行三个 Chapter01 测试组，覆盖合法加载、非法输入、失败原子性、GGUF 测试文件、模型结构约束和量化 payload。Qwen3 合成用例检查 Q/K Norm、共享输出权重和可选 BOS；Qwen2MoE 用例检查 Router、打包专家权重和共享专家。设置 `CHAPTER01_BUILD_DIR` 可以保留检查工具：

```bash
export CHAPTER01_BUILD_DIR=/tmp/cpp-llm-engine-chapter01-build
bash code/chapter01_tensor_loader/run.sh
"$CHAPTER01_BUILD_DIR/gguf_inspect" \
  llama.cpp/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf \
  --validate-schema
"$CHAPTER01_BUILD_DIR/gguf_first_layer_inspect" \
  llama.cpp/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
```

同一个结构检查工具也用于 Qwen3，并可按名称查看单个权重：

```bash
"$CHAPTER01_BUILD_DIR/gguf_inspect" \
  llama.cpp/models/qwen3-0.6b/Qwen3-0.6B-Q4_K_M.gguf \
  --validate-schema
"$CHAPTER01_BUILD_DIR/gguf_inspect" \
  llama.cpp/models/qwen3-0.6b/Qwen3-0.6B-Q4_K_M.gguf \
  --tensor blk.0.attn_q.weight
```

启用 llama.cpp reference adapter 后，可同时验证真实分词结果与 TinyLlama 模型结构：

```bash
bash code/chapter01_tensor_loader/run_real_tokenizer.sh
```

Qwen3 使用独立固定资产建立第二套真实 Loader 基线：

```bash
bash code/chapter01_tensor_loader/run_qwen3_baseline.sh
```

该脚本验证文件 SHA-256、`Hello Qwen3` 的固定词元、聊天模板、28 层元数据和 311 个张量。通过后可以把这些输入交给 Qwen 解码器测试。

Qwen1.5-MoE 使用独立布局验证 Router、打包专家和共享专家权重：

```bash
bash code/chapter01_tensor_loader/run_qwen15_moe_schema.sh
```

输出 `schema: qwen2moe tensors valid` 表示权重名称和形状通过检查。真实 FFN 执行由 Chapter04 的模型适配入口继续验证。

DeepSeek-V2-Lite 使用固定 GGUF 验证 MLA 元数据、第一层稠密 FFN 和后续 MoE 层：

```bash
bash code/chapter01_tensor_loader/run_deepseek_v2_lite_schema.sh
```

脚本检查 27 层、377 个张量、64 选 6 专家和压缩 K/V 维度。默认测试还验证组合 `attn_kv_b` 与拆分 K-B/V-B 布局，并确认 Q2_K、IQ4_NL 原始块能分别进入稠密层和 MoE 层权重对象。`code/model` 使用这些层输入构建完整的 27 层 CPU 解码器。

## 扩展入口

本章输出模型配置、张量布局和分层权重对象。`DenseDecoderTensorLayout` 保存 Llama/Qwen3 稠密布局，`MoeDecoderTensorLayout` 保存 Qwen2MoE 专家布局，`DeepSeek2DecoderTensorLayout` 保存 MLA 与稠密/MoE 混合布局。各模型在 CPU、CUDA、生成和服务层的接入状态统一见[模型支持状态](../../source/reference/model_support.md)。

| 方向 | 后续入口 |
| --- | --- |
| 真实 Decoder | [`code/model`](../model) 与 [Chapter04](../../source/chapter04_attention/06_real_tinyllama_decoder.md) |
| Q2_K/Q3_K/Q4_K/Q5_0/Q5_1/Q5_K/Q6_K/Q8_0/IQ4_NL 数值执行 | [`code/chapter05_quantization`](../chapter05_quantization) |
| CUDA 权重上传 | [`code/cuda`](../cuda) |
| 多模态 mmproj | [`code/multimodal`](../multimodal) |
| 其他模型支持 | [模型支持状态](../../source/reference/model_support.md) |
