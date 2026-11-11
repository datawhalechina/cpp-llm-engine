# Chapter 01 代码导航：模型加载与输入对象

本目录把教学格式或 GGUF 中的配置、分词信息和权重转换为教程内部对象；学习原理与顺序见 [Chapter01 导读](../chapter01_tensor_loader/intro.md)。

## 代码架构

```text
CIE1 -> ModelLoader -----------------------------> Tensor map
GGUF -> Reader -> metadata/index -> adapters -----+-> ModelBuilder
                                -> TensorLoader --+   |
                                                     +-> LoadedModel / DecoderInput
```

Reader 解析外部结构，Adapter 转换配置，Tensor Loader 读取 payload，Model Builder 检查并提交完整模型。Decoder 计算不属于本目录。

## 阅读顺序与资产

| 顺序 | 代码入口 | 观察重点 |
| --- | --- | --- |
| 1 | <a href="https://github.com/datawhalechina/cpp-llm-engine/blob/main/code/chapter01_tensor_loader/src/tensor.h"><code>tensor.h</code></a>、<a href="https://github.com/datawhalechina/cpp-llm-engine/blob/main/code/chapter01_tensor_loader/src/tensor.cpp"><code>tensor.cpp</code></a> | shape、dtype、storage 和类型化访问检查 |
| 2 | <a href="https://github.com/datawhalechina/cpp-llm-engine/blob/main/code/chapter01_tensor_loader/src/model_loader.cpp"><code>model_loader.cpp</code></a> | CIE1 记录校验与失败原子性 |
| 3 | <a href="https://github.com/datawhalechina/cpp-llm-engine/blob/main/code/chapter01_tensor_loader/src/gguf_reader.cpp"><code>gguf_reader.cpp</code></a>、<a href="https://github.com/datawhalechina/cpp-llm-engine/blob/main/code/chapter01_tensor_loader/src/gguf_tensor_loader.cpp"><code>gguf_tensor_loader.cpp</code></a> | 索引解析与 payload 读取分离 |
| 4 | <a href="https://github.com/datawhalechina/cpp-llm-engine/blob/main/code/chapter01_tensor_loader/src/model_metadata_adapter.cpp"><code>model_metadata_adapter.cpp</code></a>、<a href="https://github.com/datawhalechina/cpp-llm-engine/blob/main/code/chapter01_tensor_loader/src/tokenizer_metadata.cpp"><code>tokenizer_metadata.cpp</code></a> | 外部字段到内部契约 |
| 5 | <a href="https://github.com/datawhalechina/cpp-llm-engine/blob/main/code/chapter01_tensor_loader/src/model_builder.cpp"><code>model_builder.cpp</code></a> | schema 检查与模型装配 |

## 运行与验证

```bash
bash code/chapter01_tensor_loader/run.sh
```

三个专项测试分别覆盖 Loader、Schema、Tokenizer/Payload。已有 TinyLlama 与 llama.cpp 构建时，可继续运行：

```bash
bash code/chapter01_tensor_loader/run_real_tokenizer.sh
```

## 扩展入口

真实 Decoder 从 [`code/model`](./model.md) 继续，量化 payload 的数值计算进入 [Chapter05](../chapter05_quantization/intro.md)，模型支持范围见 [模型支持状态](../reference/model_support.md)。
