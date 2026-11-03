# chapter05_quantization 代码导航

本章代码包含 INT8、Q4_K、Q6_K 解码、量化 Linear、benchmark 和精度测试。

## 阅读顺序

1. 先看 `tensor.cpp` 和 `dequantize_op.cpp`
2. 再看 `linear_layer.cpp`、`quantized_linear.cpp` 和 `quantized_linear_policy.cpp`
3. 最后看 policy 测试、真实 Decoder 调用和 CUDA 非融合基线

## 量化路径

| 文件 | 作用 |
| --- | --- |
| <a href="/_code/chapter05_quantization/include/chapter05_quantization/tensor.h"><code>tensor.h</code></a> | 量化 Tensor 接口 |
| <a href="/_code/chapter05_quantization/src/tensor.cpp"><code>src/tensor.cpp</code></a> | 量化 Tensor 实现 |
| <a href="/_code/chapter05_quantization/src/dequantize_op.cpp"><code>src/dequantize_op.cpp</code></a> | 反量化算子 |
| <a href="/_code/chapter05_quantization/src/linear_layer.cpp"><code>src/linear_layer.cpp</code></a> | FP baseline Linear |
| <a href="/_code/chapter05_quantization/src/quantized_linear.cpp"><code>src/quantized_linear.cpp</code></a> | 量化 Linear |
| <a href="/_code/chapter05_quantization/include/chapter05_quantization/q4k_dequantize.h"><code>q4k_dequantize.h</code></a> | Q4_K block 解码接口 |
| <a href="/_code/chapter05_quantization/src/q4k_dequantize.cpp"><code>src/q4k_dequantize.cpp</code></a> | Q4_K CPU reference 反量化 |
| <a href="/_code/chapter05_quantization/src/q6k_dequantize.cpp"><code>src/q6k_dequantize.cpp</code></a> | Q6_K CPU reference 反量化 |
| <a href="/_code/chapter05_quantization/src/q6k_linear.cpp"><code>src/q6k_linear.cpp</code></a> | Q6_K reference、scalar fused 和能力探测 |
| <a href="/_code/chapter05_quantization/src/q6k_linear_avx2.cpp"><code>src/q6k_linear_avx2.cpp</code></a> | Q6_K fused AVX2/FMA backend |
| <a href="/_code/chapter05_quantization/include/chapter05_quantization/q4k_linear.h"><code>q4k_linear.h</code></a> | Q4_K Linear reference 接口 |
| <a href="/_code/chapter05_quantization/src/q4k_linear.cpp"><code>src/q4k_linear.cpp</code></a> | Q4_K reference、scalar fused 和能力探测 |
| <a href="/_code/chapter05_quantization/src/q4k_linear_avx2.cpp"><code>src/q4k_linear_avx2.cpp</code></a> | Q4_K fused AVX2/FMA backend |
| <a href="/_code/chapter05_quantization/include/chapter05_quantization/quantized_linear_policy.h"><code>quantized_linear_policy.h</code></a> | Q4_K/Q6_K backend policy 与 registry 契约 |
| <a href="/_code/chapter05_quantization/src/quantized_linear_policy.cpp"><code>src/quantized_linear_policy.cpp</code></a> | 按 weight format 统一选择并调用 backend |
| <a href="/_code/cuda/include/cpp_llm_engine/cuda/cuda_quantized_matrix.h"><code>cuda_quantized_matrix.h</code></a> | packed Q4_K/Q6_K device 权重 |
| <a href="/_code/cuda/src/quantized_dequantize_backend.cu"><code>quantized_dequantize_backend.cu</code></a> | CUDA Q4_K/Q6_K 反量化 kernel |

## 验证与脚本

| 文件 | 作用 |
| --- | --- |
| <a href="/_code/chapter05_quantization/examples/main_quantization.cpp"><code>examples/main_quantization.cpp</code></a> | 量化示例 |
| <a href="/_code/chapter05_quantization/examples/q4k_benchmark.cpp"><code>examples/q4k_benchmark.cpp</code></a> | Q4_K decode + Linear 基线 |
| <a href="/_code/chapter05_quantization/tests/chapter05_quantization_test.cpp"><code>tests/chapter05_quantization_test.cpp</code></a> | 量化格式、数值误差、后端能力和策略分派测试 |
| <a href="/_code/chapter05_quantization/tests/chapter02_tensor_quantized_linear_integration_test.cpp"><code>tests/chapter02_tensor_quantized_linear_integration_test.cpp</code></a> | Chapter02 Tensor View 到量化 Linear 的跨章契约测试 |
| <a href="/_code/cuda/tests/cuda_quantized_linear_test.cu"><code>cuda_quantized_linear_test.cu</code></a> | CPU/CUDA golden、非融合 SGEMM 和计时 |
| <a href="/_code/chapter05_quantization/scripts/run.sh"><code>scripts/run.sh</code></a> | 运行脚本 |

## 对应文档

- [chapter05 导读](../chapter05_quantization/intro.md)
