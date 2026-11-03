# 05.3 接收已解析的量化权重

Chapter01 已负责读取 GGUF 目录、模型元数据和 tensor payload。Chapter05 不再实现第二个模型加载器，而是消费解析后的名称、shape、GGML 类型和字节范围，把它们交给对应的量化解码与 Linear 路径。

## 从 GGUF 目录到计算视图

文件加载和量化计算之间需要保留四类信息：

| 信息 | 由谁提供 | Chapter05 怎样使用 |
| --- | --- | --- |
| tensor 名称 | Chapter01 GGUF 目录 | 映射到 Attention、FFN 或 LM Head 权重 |
| 逻辑 shape | GGUF tensor metadata | 检查 Linear 输入输出维度 |
| 量化类型 | GGML type | 选择 Q4_K、Q6_K 或 Q8_0 解码器 |
| payload 指针与字节数 | mmap/文件缓冲区 | 构造只读 `QuantizedMatrixView` 并检查 block 数量 |

真实 GGUF 格式把 scale、minimum 和 quant bits 放在各自 block 内，不使用教学 INT8 Tensor 的单个 `scale` 字段。执行代码必须根据量化类型计算预期字节数，拒绝 shape 与 payload 不匹配的输入。

## 所有权与执行接口

`QuantizedMatrixView` 只引用已加载的 packed 权重，不拥有模型文件。模型权重的生命周期必须覆盖 Decoder 和所有 Linear 调用；激活则通过 Chapter02 的连续 Float32 View 传入。

数据交接顺序是：

```text
GGUF 目录与 payload
    -> 名称 / shape / GGML type / 字节范围
    -> QuantizedMatrixView
    -> QuantizedLinearPolicy
    -> reference、scalar fused、AVX2 或 CUDA backend
```

这条链路避免复制完整 packed 权重，也避免让量化 backend 重新解析 GGUF 文件。文件格式错误在 Chapter01 报告，block 长度或 Linear shape 错误在 Chapter05 报告。

## 验证与衔接

默认脚本使用固定 payload 验证格式分派和错误路径：

```bash
bash code/chapter05_quantization/scripts/run.sh
```

真实 TinyLlama 链路再执行：

```bash
bash scripts/run_real_decoder_learning.sh
```

通过条件是量化类型选择正确、payload 长度匹配、模型权重生命周期有效，错误输入不会覆盖输出。下一篇 [`05.4 反量化算子`](./04_dequantize_op.md) 将逐 block 解释 packed 字节怎样恢复浮点参考值。
