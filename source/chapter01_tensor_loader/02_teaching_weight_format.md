# 01.2 教学权重格式

小型 `CIE1` 格式展示磁盘字节怎样恢复成 Tensor。它只作为 Loader 契约的测试夹具；真实 TinyLlama 路径直接读取 GGUF。

## CIE1 文件布局

<a href="../_code/chapter01_tensor_loader/src/model_loader.cpp"><code>model_loader.cpp</code></a> 按照测试数据写入器的字段顺序读取数据：

| 区域 | 字段 | 作用 |
| --- | --- | --- |
| 文件头 | `magic=CIE1`、version、tensor count | 识别格式并选择解析规则 |
| 张量记录 | 名称、数据类型、设备、维数、形状 | 恢复张量元数据 |
| Payload | byte count、raw bytes | 恢复实际权重数据 |

这个最小格式把关键契约显式摆出来：导出端写入的字段顺序、整数宽度和数据类型编号，必须与 C++ 加载器完全一致。固定测试数据直接在 C++ 中生成，因此可以同时覆盖成功路径，以及文件截断、错误魔数等失败路径。

训练生态通常用 Python 读取 Hugging Face 检查点，再写入目标格式。若为本教程补充转换器，它应复用这份格式契约，并验证张量数量、名称、形状、数据类型和权重字节数；它不参与 C++ 运行时。当前真实模型不经过该转换，而是直接读取 GGUF。

## 从教学格式到 GGUF

两条路径最终都产生内部 Tensor，但复杂度不同：

| 教学路径 | 真实路径 |
| --- | --- |
| 测试数据写入器 → CIE1 → `ModelLoader` | TinyLlama GGUF → `GgufReader` → 适配器 / `GgufTensorLoader` |
| 连续张量记录，字段很少 | 模型元数据、张量索引、数据对齐和量化类型 |
| 用于理解格式与错误处理 | 用于真实 tokenizer、权重和 Decoder 验证 |

## 验证与衔接

运行 `bash code/chapter01_tensor_loader/run.sh`，确认合法测试文件能恢复两个张量；再检查错误魔数、错误版本和截断权重数据均返回明确错误。完成后继续阅读 [`01.3 模型加载与验证`](./03_model_loader_validation.md)，观察磁盘记录如何成为可查询对象。

