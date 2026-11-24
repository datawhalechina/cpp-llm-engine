# 05.3 接收已解析的量化权重

Chapter01 读取 GGUF 目录、模型元数据和张量载荷。Chapter05 接收解析后的名称、形状、GGML 类型和字节范围，把它们交给对应的量化解码与线性层路径。

## 从 GGUF 目录到计算视图

文件加载和量化计算之间需要保留四类信息：

| 信息 | 由谁提供 | Chapter05 怎样使用 |
| --- | --- | --- |
| 张量名称 | Chapter01 GGUF 目录 | 映射到注意力、FFN 或 LM Head 权重 |
| 逻辑形状 | GGUF 张量元数据 | 检查线性层输入输出维度 |
| 量化类型 | GGML type | 选择 Q3_K、Q4_K、Q5_1、Q6_K 或 Q8_0 解码器 |
| 载荷指针与字节数 | mmap/文件缓冲区 | 构造只读 `QuantizedMatrixView` 并检查分块数量 |

真实 GGUF 格式把 scale、minimum 和 quant bits 放在各自 block 内，不使用教学 INT8 Tensor 的单个 `scale` 字段。执行代码必须根据量化类型计算预期字节数，拒绝 shape 与 payload 不匹配的输入。

## 所有权与执行接口

`QuantizedMatrixView` 借用已加载的打包权重，模型文件由上层模型对象持有。该模型对象的生命周期需要覆盖解码器和所有线性层调用；激活通过 Chapter02 的连续 Float32 视图传入。

数据交接顺序是：

```text
GGUF 目录与 payload
    -> 名称 / shape / GGML type / 字节范围
    -> QuantizedMatrixView
    -> QuantizedLinearPolicy
    -> reference、scalar fused、AVX2 或 CUDA backend
```

这条链路复用打包权重，量化计算后端直接接收解析结果。文件格式错误在 Chapter01 报告，分块长度或线性层形状错误在 Chapter05 报告。

## 验证与衔接

默认脚本使用固定 payload 验证格式分派和错误路径：

```bash
bash code/chapter05_quantization/scripts/run.sh
```

真实 TinyLlama 链路再执行：

```bash
bash code/chapter05_quantization/scripts/run_real_decoder.sh
```

通过条件是量化类型选择正确、载荷长度匹配、模型权重生命周期有效，错误输入会保留原输出。专属脚本运行固定 TinyLlama 提示词，检查上下文隔离、嵌入预填充、llama.cpp 数值指标和贪心序列；生成与多请求批处理由全局回归继续验证。下一篇 [`05.4 反量化算子`](./04_dequantize_op.md) 将逐分块解释打包字节怎样恢复浮点参考值。
