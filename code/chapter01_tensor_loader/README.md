# Chapter 01：模型加载与输入对象

本目录把教学格式或 GGUF 中的配置、分词信息和权重转换为教程内部对象。完整学习顺序见 [`source/chapter01_tensor_loader`](../../source/chapter01_tensor_loader/intro.md)。

## 代码架构

教学链使用小型 CIE1 文件展示一条完整加载过程；真实链把 GGUF 的结构解析、payload 读取和模型装配分开。两条路径最终都输出可检查的 Tensor 与模型配置，不在本章执行 Decoder。

```text
CIE1 bytes -> ModelLoader -----------------------> Tensor map

GGUF bytes -> GgufReader ------> metadata/index
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
| `src/gguf_reader.*` | 读取 GGUF header、metadata 和张量索引 |
| `src/model_metadata*`、`src/tokenizer_metadata*` | 把 GGUF 字段转换为内部配置 |
| `src/gguf_tensor_loader.*` | 按索引读取 F32/F16/I8/Q8_0/Q4_K/Q6_K payload |
| `src/model_schema.*` | 检查 Llama 权重名称与 shape |
| `src/model_builder.*` | 装配 `LoadedModel` 和 `DecoderInput` |
| `src/real_layer_weights.*` | 按层取得后续 Decoder 使用的权重 |
| `examples/gguf_inspect.cpp` | 检查真实 GGUF 配置、索引和 schema |
| `examples/gguf_first_layer_inspect.cpp` | 检查首层九个权重的 dtype、shape 和 payload |
| `tests/chapter01_*_test.cpp` | 分别验证 Loader、Schema、Tokenizer/Payload 契约 |

## 运行与验证

从仓库根目录运行默认测试：

```bash
bash code/chapter01_tensor_loader/run.sh
```

脚本隔离构建并运行三个 Chapter01 测试组，覆盖合法加载、非法输入、失败原子性、GGUF fixture、模型 schema 和量化 payload。设置 `CHAPTER01_BUILD_DIR` 可以保留检查工具：

```bash
export CHAPTER01_BUILD_DIR=/tmp/cpp-llm-engine-chapter01-build
bash code/chapter01_tensor_loader/run.sh
"$CHAPTER01_BUILD_DIR/gguf_inspect" \
  llama.cpp/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf \
  --validate-schema
"$CHAPTER01_BUILD_DIR/gguf_first_layer_inspect" \
  llama.cpp/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
```

启用 llama.cpp reference adapter 后，可同时验证真实分词结果与 TinyLlama 模型结构：

```bash
bash code/chapter01_tensor_loader/run_real_tokenizer.sh
```

## 扩展入口

本章在 `LoadedModel` 和 `DecoderInput` 处结束。后续扩展从以下位置继续：

| 方向 | 后续入口 |
| --- | --- |
| 真实 Decoder | [`code/model`](../model) 与 [Chapter04](../../source/chapter04_attention/06_real_tinyllama_decoder.md) |
| Q4_K/Q6_K/Q8_0 数值执行 | [`code/chapter05_quantization`](../chapter05_quantization) |
| CUDA 权重上传 | [`code/cuda`](../cuda) |
| 多模态 mmproj | [`code/multimodal`](../multimodal) |
| 其他模型支持 | [模型支持状态](../../source/reference/model_support.md) |
