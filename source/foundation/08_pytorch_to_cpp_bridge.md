# F08 从 PyTorch 概念映射到 C++

PyTorch 是训练和实验的参考；C++ 推理引擎需要把其中隐含的张量、模块、权重和执行顺序显式拆出来。映射目标是保持语义，而不是寻找一组同名 API。

## 从隐式语义到显式对象

| PyTorch 概念 | 教程中的对象 | 需要显式处理 |
| --- | --- | --- |
| `torch.Tensor` | `Tensor` | 存储、形状、数据类型、设备和视图 |
| `nn.Module` | 算子、层或运行时单元 | 参数所有权和 `forward` 顺序 |
| `state_dict` | `ModelLoader` / `ModelBuilder` | 名称、形状、数据类型和缺失检查 |
| `view` | 张量视图 | 偏移、步长、别名和生命周期 |
| `forward` | 运行时步骤 | 输入、状态、输出和错误路径 |
| 自动微分（autograd） | 推理只执行前向计算 | 保留前向计算所需状态 |

## 三类实现证据

| 证据 | 回答的问题 | 示例 |
| --- | --- | --- |
| 参考实现 | 小输入算得对吗 | CPU 矩阵乘法、注意力 |
| 真实基线 | 真实模型能运行吗 | llama.cpp、GGUF |
| 执行后端 | 加速路径是否一致 | CUDA、cuBLAS、量化计算核 |

`state_dict -> ModelLoader -> ModelConfig` 是 Chapter01 的输入边界；`Tensor -> Operator -> Runtime` 是后续执行边界。

## 最小示例

```python
y = torch.matmul(x, weight) + bias
```

在 C++ 中需要明确成：读取 `x/weight/bias` 的张量布局，检查形状，选择 CPU 或 CUDA 后端，写入输出张量，并由调用者管理生命周期。表面上都是一次 `forward`，实现责任已经展开。

## 验证与练习

验证时分别说明语义映射和实现差异：张量是否共享底层存储？权重如何按名称加载？数据类型、设备和视图如何落地？对应练习见 [Foundation 正文与练习映射](./exercise_map.md)，完成 FQ08；完整 `state_dict -> ModelLoader` 题目由 Chapter01 实现和测试承接。
