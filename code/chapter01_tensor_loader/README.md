# Chapter 01：模型加载与输入对象

本目录把教学格式或 GGUF 中的配置、分词信息和权重转换为教程内部对象。完整学习顺序见 [`source/chapter01_tensor_loader`](../../source/chapter01_tensor_loader/intro.md)。

## 代码架构

教学链使用小型 CIE1 文件展示一条完整加载过程；真实链把 GGUF 的结构解析、payload 读取和模型装配分开。两条路径最终都输出可检查的 Tensor 与模型配置，不在本章执行 Decoder。

```text
CIE1 bytes -> ModelLoader -----------------------> Tensor map

GGUF bytes -> GgufReader ------> 元数据/索引
                          |              |
                          |              +-> Metadata/Tokenizer adapters
                          |              +-> GgufTensorLoader -> Tensor map
                          |                                   |
                          +----------------------------> ModelBuilder
                                                           |
                                             LoadedModel + DecoderInput
```

Reader 只解释外部格式，Adapter 把外部字段转换为稳定配置，Loader 负责 payload 边界，Builder 检查模型结构。任一步失败都不提交半加载结果。

## 阅读顺序与资产

先读 `tensor.*` 和 `model_loader.*`，用小文件理解对象与失败路径；再沿 GGUF Reader、Adapter、Tensor Loader、Model Builder 阅读真实加载链；最后查看检查工具和测试。

| 资产 | 作用 |
| --- | --- |
| `src/tensor.*` | 保存名称、shape、dtype、storage 和 payload |
| `src/model_loader.*` | 加载 CIE1 教学格式并验证记录完整性 |
| `src/gguf_reader.*` | 读取 GGUF 文件头、模型元数据和张量索引 |
| `src/model_metadata*`、`src/tokenizer_metadata*` | 把 GGUF 字段转换为内部配置 |
| `src/gguf_tensor_loader.*` | 按索引读取 F32/F16/I8 与 Q3_K/Q4_K/Q5_1/Q6_K/Q8_0 payload |
| `src/model_schema.*` | 检查 Llama/Qwen3 稠密布局、Qwen2MoE 专家布局与 DeepSeek2 MLA 混合布局 |
| `src/real_layer_weights.*` | 按层装配稠密层、Qwen MoE 层和 DeepSeek2 稠密/MoE 层权重 |
| `src/model_builder.*` | 装配 `LoadedModel` 和 `DecoderInput` |
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

该脚本验证文件 SHA-256、`Hello Qwen3` 的固定词元、聊天模板、28 层元数据和 311 个张量。它不调用 Qwen Decoder，因此通过条件只表示 Chapter01 输入契约成立。

Qwen1.5-MoE 使用独立布局验证 Router、打包专家和共享专家权重：

```bash
bash code/chapter01_tensor_loader/run_qwen15_moe_schema.sh
```

输出 `schema: qwen2moe tensors valid` 表示文件结构已经进入稳定输入契约。真实 FFN 执行由 Chapter04 的模型适配入口继续验证。

DeepSeek-V2-Lite 使用固定 GGUF 验证 MLA 元数据、第一层稠密 FFN 和后续 MoE 层：

```bash
bash code/chapter01_tensor_loader/run_deepseek_v2_lite_schema.sh
```

脚本检查 27 层、377 个张量、64 选 6 专家和压缩 K/V 维度。默认测试还验证旧版组合 `attn_kv_b` 与新版拆分 K-B/V-B 布局，并保证 Q2_K、IQ4_NL 原始 block 能分别进入稠密层和 MoE 层权重对象。Loader 交付的两类层输入由 `code/model` 连接为完整 27 层 CPU Decoder；Chapter01 本身仍只负责解析、校验和权重装配。

## 扩展入口

本章在 `LoadedModel` 和 `DecoderInput` 处结束。后续扩展从以下位置继续：

`DenseDecoderTensorLayout` 保存 Llama/Qwen3 稠密布局，`MoeDecoderTensorLayout` 保存 Qwen2MoE 专家布局，`DeepSeek2DecoderTensorLayout` 保存 MLA 与混合 Dense/MoE 布局；`LlamaTensorLayout` 暂时作为兼容别名保留。Llama 与 Qwen3 已进入完整 CPU/CUDA Decoder、Generation 和服务验证；Qwen2MoE 已进入 CPU 24 层 Decoder、Generation、按专家分组批执行、Chapter08 调度和 Chapter10 服务验证。DeepSeek2 已进入 CPU 27 层 Decoder、Generation、跨请求专家分组、Chapter08 调度和 Chapter10 HTTP/SSE 验证；Chapter01 仍只负责交付其元数据、张量布局与分层权重对象。

| 方向 | 后续入口 |
| --- | --- |
| 真实 Decoder | [`code/model`](../model) 与 [Chapter04](../../source/chapter04_attention/06_real_tinyllama_decoder.md) |
| Q3_K/Q4_K/Q5_1/Q6_K/Q8_0 数值执行 | [`code/chapter05_quantization`](../chapter05_quantization) |
| CUDA 权重上传 | [`code/cuda`](../cuda) |
| 多模态 mmproj | [`code/multimodal`](../multimodal) |
| 其他模型支持 | [模型支持状态](../../source/reference/model_support.md) |
