# 03.1 从算子入口发起一次计算

矩阵乘法会出现在 Q/K/V 投影、Attention 输出和 FFN 中。如果每个调用位置都直接写三重循环，后续切换计算库时就要逐处修改。Chapter03 因此保留一个调用入口，把“要计算哪两个矩阵”和“用什么方法计算”分开。

## 为什么保留统一调用入口

<a href="../_code/chapter03_matmul/include/chapter03_matmul/operator.h"><code>operator.h</code></a> 定义双输入、单输出接口：

```cpp
class Operator {
public:
    virtual ~Operator() = default;
    virtual Matrix compute(const Matrix& lhs, const Matrix& rhs) const = 0;
};
```

`MatMulOp` 实现 `compute()`。调用者传入两个 `Matrix`，得到一个新的输出 `Matrix`；CPU 循环由 `CpuReferenceMatmulBackend` 执行。选读 GPU 路线时，[`03.4`](./04_cublas_implement.md) 会把相同的矩阵形状交给设备指针和 cuBLAS。

多一层调用接口会增加一个类和一次分派，但换来两个直接收益：调用方只依赖计算语义，CPU 与 CUDA 可以分别测试。Chapter04 采用独立的教学 Linear，并复用本章的形状约束、乘加语义和数值基线。学习者仍然能够沿着 `MatMulOp → Backend → Kernel` 找到本章的实际计算代码。

## Matrix 怎样连接 Chapter02 的视图

`Matrix` 适合用小型连续数据观察矩阵乘法，Chapter02 的 `Tensor` 则负责共享存储、切片和转置。两者通过形状、步长、偏移和存储范围连接起来。

| 对象 | 数据放在哪里 | 本章怎样使用 |
| --- | --- | --- |
| `Matrix` | 自己持有连续的 `vector<float>` | 运行示例和 CPU 三重循环 |
| `TensorViewRef` | 引用 Chapter02 的共享存储 | 作为只读矩阵乘法输入 |
| `MutableTensorViewRef` | 引用可写共享存储 | 接收矩阵乘法输出 |
| `TensorContract` | 保存形状、步长、偏移、类型和设备 | 检查视图是否能安全参与计算 |

<a href="../_code/chapter03_matmul/src/tensor_matmul_adapter.cpp"><code>tensor_matmul_adapter.cpp</code></a> 把视图首地址和二维形状整理成矩阵乘法请求。这样可以先用 `Matrix` 学会算法，再用同一套地址检查处理 Chapter02 的真实视图。

## 动手验证

运行下面的命令：

```bash
bash code/chapter03_matmul/scripts/run.sh
```

示例应输出一个 `2x2` 结果，CTest 同时检查 `Operator` 调用、`Matrix::contract()` 和 Chapter02 视图接入。接下来阅读 [`03.2`](./02_matmul_design.md)，看一次请求从形状检查走到结果写回。
