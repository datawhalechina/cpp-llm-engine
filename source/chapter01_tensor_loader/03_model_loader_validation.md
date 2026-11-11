# 01.3 模型加载与验证

`ModelLoader` 把 CIE1 文件恢复成按名称查询的张量集合。加载过程必须在数据进入算子前暴露格式、名称、形状和权重数据错误。

## 从文件恢复 Tensor

<a href="https://github.com/datawhalechina/cpp-llm-engine/blob/main/code/chapter01_tensor_loader/src/model_loader.cpp"><code>model_loader.cpp</code></a> 中的 `load()` 每次先清空旧状态，再依次校验魔数、版本和张量数量。每条记录恢复名称、数据类型、设备、形状和权重数据，先放入局部 Tensor Map；只有全部记录成功后才提交结果，失败时不会暴露半加载模型。

## 在最早位置拒绝错误

加载完成表示文件已经变成运行时可访问对象，但还需要检查对象是否符合模型 schema。排查顺序应从最早的确定性证据开始：

1. 文件能否打开，magic 和 version 是否匹配。
2. 张量记录是否完整，名称是否重复，权重数据是否截断。
3. 数据类型和设备是否受支持，形状是否为正且不溢出。
4. 形状、数据类型和 payload 字节数是否严格一致。
5. 模型 schema 是否能找到必需权重。
6. 算子输入是否符合已验证的对象要求。

这个顺序可以避免把加载错误带进 Attention 或 MatMul 后再定位。

打印张量名称、形状、数据类型和存储大小适合快速观察；自动化检查负责固定长期行为。测试至少覆盖：

- 合法文件恢复完整 tensor map
- 按名称命中已知 tensor，未知名称返回空
- 形状与权重数据可还原为预期数值
- 错误 magic、version、dtype、device、shape 和截断输入被拒绝
- 重复名称和与 shape 不匹配的 payload 被拒绝
- 失败后 Tensor Map 保持为空

真实 GGUF 使用同样的验证思想，但由 [`01.5 模型元数据与 GGUF`](./05_model_metadata_and_gguf.md) 继续检查模型元数据、张量索引、数据对齐、量化权重数据和 Llama 结构约束。

## 验证与衔接

```bash
bash code/chapter01_tensor_loader/run.sh
```

通过标准是 CTest 全部成功，并且失败用例在加载阶段返回可定位的错误。不要用后续算子的数值结果替代加载器的名称、形状和权重数据检查。

教学格式验证通过后，[`模型元数据与 GGUF`](./05_model_metadata_and_gguf.md) 使用相同检查顺序处理真实模型文件。

