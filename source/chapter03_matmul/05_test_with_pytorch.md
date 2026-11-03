# 03.5 用 PyTorch 独立复算

独立数值参考可以帮助判断错误来自数学定义还是计算后端映射。PyTorch 适合生成小矩阵基准结果，但不进入本教程默认 CMake 和 CTest 依赖。

## 用固定输入复算结果

下面的输入与 Chapter03 C++ 测试一致，可以直接复算 `58/64/139/154`：

```python
import torch

lhs = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32)
rhs = torch.tensor([[7, 8], [9, 10], [11, 12]], dtype=torch.float32)
expected = torch.tensor([[58, 64], [139, 154]], dtype=torch.float32)
actual = lhs @ rhs

torch.testing.assert_close(actual, expected)
print(actual)
```

这段代码是学习时可选的独立复算，不属于默认回归。需要把对照扩展成固定测试数据时，再记录输入、输出、数据类型和误差容限；C++ 测试只读取结果，不在默认回归中现场安装或调用 PyTorch。

## 当前仓库怎样完成对照

当前 Chapter03 CTest 使用固定小矩阵和手算结果验证 CPU 参考实现；CUDA 的 cuBLAS 测试使用相同形状的 CPU 三重循环生成基准结果。仓库尚未提交独立 PyTorch 测试数据，因此不能把 PyTorch 对照写成已执行的默认测试。

判断顺序是：CPU 参考实现先通过固定结果，其他计算后端再与 CPU 基准结果比较。两者都偏离独立测试数据时，才回到数学定义、输入布局或测试数据本身排查。

## 验证与衔接

运行 `bash code/chapter03_matmul/scripts/run.sh` 验证 CPU 结果；具备 CUDA 环境时运行 `CUDA_HOME=/usr/local/cuda-12.8 bash code/cuda/scripts/run.sh` 验证 cuBLAS 与 CPU 基准结果。需要增加 PyTorch 测试数据时，应固定随机种子、形状、数据类型和容差，并保持默认 CTest 不依赖 Python 环境。
