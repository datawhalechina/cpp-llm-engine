# 03.0 构建矩阵乘法算子与计算后端

## 本章导读

Chapter02 已经能用张量视图找到一段数据。本章把两个连续二维视图送入矩阵乘法，先用 CPU 三重循环算出可手工复查的结果，再观察同一次计算怎样交给 CPU Backend 或可选的 cuBLAS。Chapter04 将用这些矩阵乘法组装线性层、注意力和前馈网络。

在 Transformer 中，一组词元特征可以写成 `X[T,D]`：`T` 是词元数，`D` 是每个词元的特征宽度。线性投影用 `X[T,D] × W[D,O]` 得到 `Y[T,O]`，Q/K/V 投影、Attention 输出投影和 FFN 都会重复这项计算。Chapter03 先用一个二维例子把形状、地址和计算方法讲清楚。

![Chapter 03 主图：矩阵乘法与计算后端](../assets/diagrams/chapter03-main.svg)

## 开始学习

先运行本章脚本：

```bash
bash code/chapter03_matmul/scripts/run.sh
```

脚本先打印矩阵乘法结果，再运行 CTest。看到 `value(0,0)=58`、`value(1,1)=154` 和全部测试通过后，按 CPU 主线继续阅读；有 NVIDIA GPU 时再进入 G 扩展。

| 路线 | 阅读顺序 | 先看代码 | 完成后能解释什么 |
| --- | --- | --- | --- |
| A/B CPU 主线 | [`03.1`](./01_operator_base.md) → [`03.2`](./02_matmul_design.md) → [`03.3`](./03_cpu_reference.md) → [`03.5`](./05_test_with_pytorch.md) → [`03.6`](./06_complexity_intuition.md) → [`03.7`](./07_runtime_backend_bridge.md) | <a href="../_code/chapter03_matmul/src/cpu_kernel.cpp"><code>cpu_kernel.cpp</code></a>、<a href="../_code/chapter03_matmul/src/tensor_matmul_adapter.cpp"><code>tensor_matmul_adapter.cpp</code></a> | 张量视图怎样进入 CPU 循环并安全写回结果 |
| G GPU 扩展 | 完成 03.3 后选读 [`03.4`](./04_cublas_implement.md) | <a href="../_code/cuda/src/cublas_matmul_backend.cpp"><code>cublas_matmul_backend.cpp</code></a> | 行优先矩阵怎样交给 cuBLAS，并与 CPU 结果比较 |
| M 多模态复用 | 完成 CPU 主线后查看[多模态学习指南](../guides/multimodal_inference.md) | <a href="../_code/multimodal/src/vision_projector.cpp"><code>vision_projector.cpp</code></a> | 视觉线性层和投影层怎样复用矩阵乘法 |

## 验证与下一步

CPU 主线完成后进入 [`Chapter 04`](../chapter04_attention/intro.md)，把矩阵乘法放进线性层、注意力和前馈网络。下面两条命令分别检查默认 CPU 路径和可选 CUDA 路径。

| 路径 | 执行入口 | 通过条件 |
| --- | --- | --- |
| CPU 必做 | `bash code/chapter03_matmul/scripts/run.sh` | 固定结果、连续视图、失败后不覆盖输出全部通过 |
| CUDA 选做 | `CUDA_HOME=/usr/local/cuda-12.8 bash code/cuda/scripts/run.sh` | cuBLAS 与 CPU 三重循环的结果在容差内一致 |
