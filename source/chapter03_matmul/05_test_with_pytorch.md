# 03.5 用 PyTorch 复算同一组矩阵

手算可以检查很小的例子，PyTorch 则提供另一份现成实现。把同一组输入交给 C++ 和 PyTorch，可以先确认期望结果，再排查 CPU 循环或 CUDA 参数。

## 复算固定输入

下面的输入与 Chapter03 C++ 示例一致：

```python
import torch

lhs = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32)
rhs = torch.tensor([[7, 8], [9, 10], [11, 12]], dtype=torch.float32)
expected = torch.tensor([[58, 64], [139, 154]], dtype=torch.float32)
actual = lhs @ rhs

torch.testing.assert_close(actual, expected)
print(actual)
```

输出应为：

```text
tensor([[ 58.,  64.],
        [139., 154.]])
```

PyTorch 是选做工具，CPU 主线不要求安装它。默认 CTest 已把同一组输入和期望结果写入 C++ 测试。

## 根据差异定位问题

| 对照结果 | 优先检查 |
| --- | --- |
| PyTorch 正确，CPU 结果错误 | 三重循环索引、`M/N/K` 和行优先地址 |
| CPU 正确，cuBLAS 结果错误 | A/B 交换、`m/n/k`、leading dimension 和设备指针 |
| 三者结果一致 | 当前形状的数学计算和布局映射成立 |

先运行 `bash code/chapter03_matmul/scripts/run.sh` 检查 CPU；有 NVIDIA GPU 时，再运行 `CUDA_HOME=/usr/local/cuda-12.8 bash code/cuda/scripts/run.sh`。CPU 与 CUDA 测试通过后，继续阅读 [`03.6`](./06_complexity_intuition.md)，比较相同算法工作量下的实际执行成本。
