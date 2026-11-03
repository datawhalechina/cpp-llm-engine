# 01.6 模型加载器框架对照

Chapter01 把外部模型文件转换为稳定的配置、分词器、张量映射和模型布局。成熟框架支持更多格式、设备和并行加载策略，但仍要完成相同的数据转换。

## 对象映射

| 教程对象 | `llama.cpp` | vLLM | 阅读重点 |
| --- | --- | --- | --- |
| `GgufReader`、`GgufTensorLoader` | GGUF 读取器、模型加载器 | 检查点加载器 | 文件索引何时转换为权重对象 |
| `ModelMetadataAdapter`、`ModelConfig` | 架构元数据、模型参数 | 模型配置、加载配置 | 外部字段怎样归一化并校验 |
| `ModelBuilder`、Llama tensor layout | model tensor creation/loading | 参数名映射和 weight loading | Tensor 名称怎样绑定到模型结构 |
| `TokenizerMetadataAdapter` | GGUF 分词器元数据 | 分词器配置 | 词表与特殊词元怎样和模型配置对齐 |

## 教学实现与生产实现

| Chapter01 可验证对象 | 成熟系统继续增加 |
| --- | --- |
| 单个 TinyLlama GGUF | SafeTensors、分片 checkpoint、远程模型仓库 |
| CPU 张量映射和显式权重数据 | mmap、异步读取、设备放置和权重缓存 |
| Llama schema 与名称映射 | 多模型架构注册、tensor parallel 切分 |
| 元数据、形状、数据类型的明确错误 | 延迟加载、分布式加载和恢复策略 |

模型权重如何被 Decoder 执行、哪些状态按请求隔离，进入 [`Chapter04 真实 TinyLlama Decoder`](../chapter04_attention/06_real_tinyllama_decoder.md) 继续学习。

## 继续阅读源码

阅读成熟项目时，可以沿以下问题定位代码：

1. Tensor 名称在什么时候从文件 schema 转换为模型参数？
2. 配置、权重索引和权重数据分别在哪一步校验？
3. 增加一种 checkpoint 格式时，哪些内部对象保持不变？

| 项目卡 | 阅读方向 |
| --- | --- |
| [llama.cpp](../reference/ecosystem/llama_cpp.md) | GGUF 读取器、模型参数和 mmap 加载 |
| [vLLM](../reference/ecosystem/vllm.md) | 多格式检查点加载器、设备放置和并行权重映射 |
