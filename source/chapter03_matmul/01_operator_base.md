# 03.1 算子基类

Chapter03 需要把稳定的算子调用方式和具体计算实现分开。当前教学接口使用主机端 `Matrix` 表达连续二维输入，并让 `MatMulOp` 把计算交给 CPU 计算后端。

## 算子固定什么

<a href="../_code/chapter03_matmul/include/chapter03_matmul/operator.h"><code>operator.h</code></a> 定义双输入、单输出接口：

```cpp
class Operator {
public:
    virtual ~Operator() = default;
    virtual Matrix compute(const Matrix& lhs, const Matrix& rhs) const = 0;
};
```

调用方只表达“用两个输入计算一个输出”。CPU 参考实现的循环和资源不会进入这个基类。cuBLAS 使用设备指针和 CUDA 流，因此通过 `code/cuda` 的独立入口执行，而不是接收这里的主机端 `Matrix`。

## 教学矩阵与公共契约

Chapter03 的 `Matrix` 拥有连续行优先 `vector<float>`，与 Chapter02 的共享视图不是同一个类。两者都能导出 `TensorContract`，因此形状、步长、偏移、数据类型和存储范围使用相同检查语义。

这种关系是公共契约复用，不是对象直接传递。当前教学用 `Matrix` 保持简单连续布局；适配器负责把通用张量视图与算子请求连接起来。

## 验证与衔接

运行 `bash code/chapter03_matmul/scripts/run.sh`，确认 `MatMulOp` 可以通过 `Operator` 调用，`Matrix::contract()` 输出 `[rows, cols]`、`[cols, 1]` 和有效存储范围。[`矩阵乘法算子设计`](./02_matmul_design.md) 将继续固定乘法语义和后端请求。
