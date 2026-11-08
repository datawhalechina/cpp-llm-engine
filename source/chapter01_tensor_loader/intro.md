# 01.0 加载模型权重并建立张量对象

## 本章导读

Chapter 00 已经用 llama.cpp 跑通 TinyLlama，证明模型文件、分词器和运行环境可用；本章开始建立教程自己的输入对象，把模型文件转换为配置、权重张量和模型对象，把用户文本转换为词元编号（token ID）。Chapter 02 会继续学习张量怎样共享存储和改变观察方式。

| 学习线 | 本章怎样学习 | 可验证产物 |
| --- | --- | --- |
| A 教学闭环 | 用小型 CIE1 文件、固定测试数据和教学分词器分开观察格式、加载过程与输入错误 | 可检查的张量、错误文件拒绝和词元编号 |
| B 真实执行闭环 | 读取 TinyLlama GGUF 的模型元数据、真实分词器信息和权重目录 | `ModelConfig`、张量映射、Llama 权重布局和真实文件检查 |
| G GPU 扩展坐标 | 复用 Host 侧 GGUF 解析结果，再由 CUDA 适配器上传权重 | 本章不在 GPU 上解析文件；设备权重验证位于 `code/cuda` |
| M 多模态扩展坐标 | 复用 GGUF Reader 读取视觉投影 metadata 与 payload | mmproj schema 与视觉权重验证位于 `code/multimodal` |

| 本章新术语 | 先这样理解 | 首次解释 |
| --- | --- | --- |
| 张量（`Tensor`） | 带名称、形状、数据类型和底层字节的数据对象 | [`01.1 Tensor Object`](./01_tensor_object.md) |
| 模型加载器（`ModelLoader`） | 把文件中的权重恢复成可按名称查询的张量 | [`01.3 ModelLoader and Validation`](./03_model_loader_validation.md) |
| 分词器（`Tokenizer`） | 在用户文本和模型使用的词元编号之间转换 | [`01.4 输入契约与分词器`](./04_input_contract_and_tokenizer.md) |
| 模型元数据（model metadata） | 描述模型层数、隐藏维度、词表和上下文长度 | [`01.5 Model Metadata and GGUF`](./05_model_metadata_and_gguf.md) |
| GGUF | 保存模型配置、分词信息、权重目录和权重数据的文件格式 | [`01.5 Model Metadata and GGUF`](./05_model_metadata_and_gguf.md) |

![Chapter 01 主图：模型文件和文本输入转换为内部对象](../assets/diagrams/chapter01-main.svg)

## 开始学习

先运行不依赖真实模型文件的默认测试：

```bash
bash code/chapter01_tensor_loader/run.sh
```

脚本会运行 Loader、Schema、Tokenizer/Payload 三个专项 CTest，检查 Tensor、教学权重格式、Tokenizer、模型配置和 GGUF fixture。真实 TinyLlama 检查需要 Chapter 00 下载的 GGUF，按 `01.5` 中的可选命令运行。

| 顺序 | 路线 | 页面 | 先看源码 | 读完后应看到什么 |
| --- | --- | --- | --- | --- |
| 1 | A/B 公共 | [`01.1 张量对象`](./01_tensor_object.md) | <a href="../_code/chapter01_tensor_loader/src/tensor.h"><code>tensor.h</code></a> | 名称、形状、类型和底层字节组成一个 Tensor |
| 2 | A | [`01.2 教学权重格式`](./02_teaching_weight_format.md)、[`01.3 模型加载与验证`](./03_model_loader_validation.md) | <a href="../_code/chapter01_tensor_loader/src/model_loader.cpp"><code>model_loader.cpp</code></a> | 小文件固定 Loader 正常与失败路径 |
| 3 | A→B | [`01.4 输入契约与分词器`](./04_input_contract_and_tokenizer.md) | <a href="../_code/chapter01_tensor_loader/src/tokenizer.h"><code>tokenizer.h</code></a> | 教学接口怎样过渡到模型规定的 tokenizer |
| 4 | B | [`01.5 模型元数据与 GGUF`](./05_model_metadata_and_gguf.md) | <a href="../_code/chapter01_tensor_loader/src/gguf_reader.cpp"><code>gguf_reader.cpp</code></a>、<a href="../_code/chapter01_tensor_loader/src/model_builder.cpp"><code>model_builder.cpp</code></a> | GGUF 配置和权重进入张量映射与 Llama 权重布局 |
| 5 | B | [`Loader 框架对照`](./06_framework_loader_comparison.md) | [`llama.cpp` 项目卡](../reference/ecosystem/llama_cpp.md) | 在成熟框架中定位格式解析、权重加载和模型构建 |

## 验证与下一步

本章留下可检查的权重张量、模型配置、分词器、张量映射和 Llama 权重布局，不执行完整解码器。[`Chapter 02`](../chapter02_tensor_ops/intro.md) 用独立的教学张量建立视图与执行布局；[`Chapter 04`](../chapter04_attention/06_real_tinyllama_decoder.md) 通过 `code/model` 消费这些资产并执行真实解码器。

| 验收层级 | 执行入口 | 通过条件 | 本章要求 |
| --- | --- | --- | --- |
| L1 默认 CPU | `bash code/chapter01_tensor_loader/run.sh` | Tensor、教学 Loader、Tokenizer、模型配置、GGUF fixture 和错误路径通过 CTest | 必做 |
| L2 真实模型 | `bash code/chapter01_tensor_loader/run_real_tokenizer.sh` | 真实 tokenizer、metadata、201 个 Tensor index、22 层名称、shape、dtype 和 payload offset 通过固定检查 | 有模型时必做 |
| L3 可选 CUDA | 不单独设置 Loader CUDA 验收 | 文件解析和权重索引保持 Host 侧职责，设备上传由后续 backend 验证 | 不适用 |
