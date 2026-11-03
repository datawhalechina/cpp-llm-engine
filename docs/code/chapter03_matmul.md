# chapter03_matmul 代码导航

本章代码包含 MatMul contract、CPU backend、reference kernel、示例和测试。

## 阅读顺序

1. 先看 `operator` / `matmul_op` 边界
2. 再看 `matrix.cpp` 的 CPU reference
3. 最后看测试和示例

## 算子边界

| 文件 | 作用 |
| --- | --- |
| <a href="/_code/chapter03_matmul/include/chapter03_matmul/operator.h"><code>operator.h</code></a> | 算子抽象 |
| <a href="/_code/chapter03_matmul/src/operator.cpp"><code>src/operator.cpp</code></a> | 算子基类实现 |
| <a href="/_code/chapter03_matmul/include/chapter03_matmul/matmul_op.h"><code>matmul_op.h</code></a> | MatMul 与 `try_compute` 接口 |
| <a href="/_code/chapter03_matmul/src/matmul_op.cpp"><code>src/matmul_op.cpp</code></a> | Operator 到 backend 的 `Status` 适配 |

## 验证与脚本

| 文件 | 作用 |
| --- | --- |
| <a href="/_code/chapter03_matmul/src/matrix.cpp"><code>src/matrix.cpp</code></a> | CPU reference |
| <a href="/_code/chapter03_matmul/examples/main_matmul.cpp"><code>examples/main_matmul.cpp</code></a> | 示例入口 |
| <a href="/_code/chapter03_matmul/tests/chapter03_matmul_test.cpp"><code>tests/chapter03_matmul_test.cpp</code></a> | Matrix 数值、边界和 Backend 失败原子性 |
| <a href="/_code/chapter03_matmul/tests/chapter02_tensor_matmul_integration_test.cpp"><code>tests/chapter02_tensor_matmul_integration_test.cpp</code></a> | Tensor View、显式连续化和跨章适配契约 |
| <a href="/_code/chapter03_matmul/scripts/run.sh"><code>scripts/run.sh</code></a> | 运行脚本 |

## 对应文档

- [chapter03 导读](../chapter03_matmul/intro.md)
