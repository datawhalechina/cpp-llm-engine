# Chapter 03：MatMul 与计算后端

本目录把主机 Matrix、CPU reference、算子接口和 backend 分派连接起来。设备内存和 cuBLAS 执行位于独立的 `code/cuda` target。完整学习说明见 [`source/chapter03_matmul`](../../source/chapter03_matmul/intro.md)。

## 本章资产总览

| 资产 | 作用 |
| --- | --- |
| `include/` | `Operator`、`MatMulOp`、backend 和请求契约 |
| `src/operator.cpp` | 算子公共行为 |
| `src/matmul_op.cpp` | MatMul 算子、`try_compute` 和 backend 请求适配 |
| `src/tensor_matmul_adapter.cpp` | 只读输入 View、可写输出 View 的布局校验与 `MatmulRequest` 转换 |
| `src/cpu_backend.cpp` | CPU reference backend 分派 |
| `src/cpu_kernel.cpp` | row-major CPU 三重循环 kernel |
| `src/matrix.cpp` | 教学矩阵存储与计算 |
| `tests/chapter03_matmul_test.cpp` | 数值正确性、shape 和边界检查 |
| `tests/operator_smoke_test.cpp` | Operator/backend 接口 smoke |
| `tests/chapter02_tensor_matmul_integration_test.cpp` | Chapter02 offset、slice/transpose 拒绝与显式连续化路径 |
| `examples/main_matmul.cpp` | demo 输出和调用示例 |
| `scripts/run.sh` | 隔离构建、CTest 和 demo 入口 |
| `code/core` | 共享 `Backend`、`Operator`、`Status` 契约 |

## 运行与验证

从仓库根目录执行：

```bash
bash code/chapter03_matmul/scripts/run.sh
```

`Matrix::contract()` 和 Chapter02 View 共享公共布局语义；适配器只接受 CPU Float32 连续 row-major View。非连续 slice 必须先显式调用 Chapter02 的 `contiguous()`，适配器本身不会隐藏输入复制。`try_compute_tensor_views_into()` 将结果写入调用方提供的可写 View；为保证 backend 失败时原输出不变，当前先在临时结果中计算，成功后提交。`compute()` 和 `Matrix` 继续保留为二维教学入口。
