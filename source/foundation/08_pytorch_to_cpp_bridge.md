# F08 从 PyTorch 概念映射到 C++

PyTorch 是训练和实验的参考；C++ 推理引擎需要把其中隐含的 Tensor、module、权重和执行顺序显式拆出来。映射目标是保持语义，不是寻找一组同名 API。

## 主线位置与学习目标

| PyTorch 概念 | 教程中的对象 | 需要显式处理 |
| --- | --- | --- |
| `torch.Tensor` | `Tensor` | storage、shape、dtype、device、view |
| `nn.Module` | operator / layer / runtime unit | 参数所有权和 `forward` 顺序 |
| `state_dict` | `ModelLoader` / `ModelBuilder` | 名称、shape、dtype 和缺失检查 |
| `view` | Tensor view | offset、stride、别名和生命周期 |
| `forward` | Runtime step | 输入、状态、输出和错误路径 |
| autograd | 不进入推理 runtime | 只保留前向计算所需状态 |

## 三类实现证据

| 证据 | 回答的问题 | 示例 |
| --- | --- | --- |
| reference | 小输入算得对吗 | CPU MatMul、Attention |
| baseline | 真实模型能跑吗 | llama.cpp、GGUF |
| backend | 加速路径是否一致 | CUDA、cuBLAS、量化 kernel |

`state_dict -> ModelLoader -> ModelConfig` 是 Chapter 01 的输入边界；`Tensor -> Operator -> Runtime` 是后续执行边界。

## 最小示例

```python
y = torch.matmul(x, weight) + bias
```

在 C++ 中需要明确成：读取 `x/weight/bias` 的 Tensor layout，检查 shape，选择 CPU 或 CUDA backend，写入输出 Tensor，并由调用者管理生命周期。表面上都是一次 `forward`，实现责任已经展开。

## 验证与练习

验证时分别说明语义映射和实现差异：Tensor 是否共享 storage？权重如何按名称加载？dtype、device 和 view 如何落地？对应练习见 [Foundation 正文与练习映射](./exercise_map.md)，完成 FQ08；完整 `state_dict -> ModelLoader` 题目由 Chapter 01 实现和测试承接。
