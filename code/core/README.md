# Core 公共接口

`code/core` 保存各章和 CPU/CUDA 实现共同使用的数据格式。它让 Tensor、算子、Decoder 和后端在不互相包含具体实现的情况下交换请求与结果。

## 代码架构

一次典型调用从拥有数据的章节对象开始，经由非拥有视图或请求结构进入后端：

```text
Chapter02 Tensor / Chapter01 权重
                │
                ▼
 TensorViewRef / LinearRequest / AttentionRequest
                │
                ▼
       CPU backend 或 CUDA backend
                │
                ▼
          Status + 调用方输出缓冲区
```

| 接口组 | 解决的问题 | 主要使用者 |
| --- | --- | --- |
| `status.h` | 让后端显式返回成功或错误信息 | 全部计算路径 |
| `tensor_contract.h`、`tensor_view_ref.h` | 描述形状、步长、偏移、设备和借用的数据地址 | Chapter02–05、CUDA |
| `matmul_contract.h`、`linear_contract.h` | 固定矩阵乘和线性层的连续行主序输入输出 | Chapter03、Chapter05、真实模型 |
| `attention_contract.h`、`gqa_attention_contract.h`、`rope_contract.h` | 固定 Attention、GQA 和 RoPE 的维度与位置语义 | Chapter04、CUDA |
| `decoder_contract.h`、`decoder_elementwise_contract.h` | 描述 Decoder 结构和单请求执行上下文 | Chapter04、Engine、CUDA |
| `quantization_contract.h` | 计算 Q4_K、Q6_K、Q8_0 权重需要的块数和字节数 | Chapter05、CUDA |

这些类型不拥有模型权重或 Tensor storage，也不加载 GGUF、执行调度或选择设备。调用方必须保证借用数据在调用期间有效；具体后端负责校验其支持的格式并执行计算。

## 阅读顺序与资产

先读 `status.h` 和 `tensor_contract.h`，理解错误传播和数据布局；再按学习章节阅读对应请求文件。需要理解完整生成时，最后阅读 `decoder_contract.h` 中的 `DecoderExecutionContext`。

`tests/core_contract_test.cpp` 独立验证布局溢出、storage 范围、量化字节数、Decoder 维度和默认批处理失败路径。各章节测试继续验证具体数学结果，避免把实现正确性误归给接口测试。

## 运行与验证

```bash
cmake -S code/core -B /tmp/cpp-llm-engine-core-build
cmake --build /tmp/cpp-llm-engine-core-build
ctest --test-dir /tmp/cpp-llm-engine-core-build --output-on-failure
```

## 扩展入口

新增 CPU 或 CUDA 实现时，优先复用现有请求结构；只有多个章节或设备确实需要同一种数据语义时，才向 `core` 增加接口。新增字段需要同时补溢出、空指针、形状错误和正常路径测试。
