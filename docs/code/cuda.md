# CUDA 公共基础层代码导航

这组代码为后续 CUDA backend 提供错误传播、内存所有权、异步执行和 Tensor 级调度，不进入 CPU 默认回归。资源和算子 backend 读完后，再看执行上下文怎样收起裸 CUDA 对象。

## 阅读顺序

1. 阅读 `cuda_status.h`，确认 CUDA 错误怎样进入公共状态边界
2. 阅读 `device_buffer.h` 和 `pinned_host_buffer.h`，区分 device memory 与页锁定 host memory
3. 阅读 `cuda_stream.h` 和 `cuda_event.h`，确认异步顺序、同步边界和设备计时
4. 阅读 `elementwise_backend.h/.cu`，观察公共请求怎样映射到 grid/block
5. 阅读 `cublas_matmul_backend.h/.cpp`，观察 row-major 请求怎样映射到 SGEMM
6. 阅读 `cuda_tensor.h`、`cuda_workspace.h` 和 `cuda_execution_context.h`
7. 阅读 `cuda_quantized_matrix.h` 和 `quantized_dequantize_backend.h/.cu`
8. 阅读 core `rope_contract.h` 和 CUDA `rope_backend.h/.cu`，核对布局与位置语义
9. 阅读 `cuda_kv_cache.h/.cpp`，确认连续 K/V 的容量、append 和请求所有权
10. 阅读 core `gqa_attention_contract.h` 和 CUDA `gqa_attention_backend.h/.cu`
11. 阅读 `decoder_elementwise_backend.h/.cu`，补齐 RMSNorm 与 SwiGLU
12. 对照测试检查 prefill/decode、head 映射、归一化和错误 shape
13. 在能访问 GPU 设备节点的宿主环境运行脚本

## 资产总览

| 资产 | 作用 |
| --- | --- |
| <a href="https://github.com/datawhalechina/cpp-llm-engine/blob/main/code/cuda/include/cpp_llm_engine/cuda/cuda_status.h"><code>cuda_status.h</code></a> | CUDA Runtime 错误契约 |
| <a href="https://github.com/datawhalechina/cpp-llm-engine/blob/main/code/cuda/include/cpp_llm_engine/cuda/device_buffer.h"><code>device_buffer.h</code></a> | move-only 设备缓冲区接口 |
| <a href="https://github.com/datawhalechina/cpp-llm-engine/blob/main/code/cuda/include/cpp_llm_engine/cuda/pinned_host_buffer.h"><code>pinned_host_buffer.h</code></a> | move-only 页锁定 host 缓冲区接口 |
| <a href="https://github.com/datawhalechina/cpp-llm-engine/blob/main/code/cuda/include/cpp_llm_engine/cuda/cuda_stream.h"><code>cuda_stream.h</code></a> | 非阻塞 stream 所有权和同步接口 |
| <a href="https://github.com/datawhalechina/cpp-llm-engine/blob/main/code/cuda/include/cpp_llm_engine/cuda/cuda_event.h"><code>cuda_event.h</code></a> | event 记录、同步和设备计时接口 |
| <a href="https://github.com/datawhalechina/cpp-llm-engine/blob/main/code/cuda/src/cuda_status.cpp"><code>cuda_status.cpp</code></a> | 错误文本与 core Status 转换 |
| <a href="https://github.com/datawhalechina/cpp-llm-engine/blob/main/code/cuda/src/device_buffer.cpp"><code>device_buffer.cpp</code></a> | 分配、同步/异步复制和释放实现 |
| <a href="https://github.com/datawhalechina/cpp-llm-engine/blob/main/code/cuda/tests/cuda_runtime_test.cu"><code>cuda_runtime_test.cu</code></a> | 真实 GPU 异步流水线、计时和失败路径测试 |
| <a href="https://github.com/datawhalechina/cpp-llm-engine/blob/main/code/cuda/include/cpp_llm_engine/cuda/elementwise_backend.h"><code>elementwise_backend.h</code></a> | 公共 VectorAdd 请求的 CUDA backend 接口 |
| <a href="https://github.com/datawhalechina/cpp-llm-engine/blob/main/code/cuda/tests/cuda_elementwise_test.cu"><code>cuda_elementwise_test.cu</code></a> | CPU golden、grid 边界和固定 workload 计时 |
| <a href="https://github.com/datawhalechina/cpp-llm-engine/blob/main/code/cuda/include/cpp_llm_engine/cuda/cublas_matmul_backend.h"><code>cublas_matmul_backend.h</code></a> | move-only handle 和 row-major SGEMM backend |
| <a href="https://github.com/datawhalechina/cpp-llm-engine/blob/main/code/cuda/tests/cuda_cublas_matmul_test.cu"><code>cuda_cublas_matmul_test.cu</code></a> | 多 shape golden、错误输入和 SGEMM 计时 |
| <a href="https://github.com/datawhalechina/cpp-llm-engine/blob/main/code/cuda/include/cpp_llm_engine/cuda/cuda_tensor.h"><code>cuda_tensor.h</code></a> | 连续 Float32 CUDA Tensor 和 contract |
| <a href="https://github.com/datawhalechina/cpp-llm-engine/blob/main/code/cuda/include/cpp_llm_engine/cuda/cuda_workspace.h"><code>cuda_workspace.h</code></a> | 可增长复用的临时显存 |
| <a href="https://github.com/datawhalechina/cpp-llm-engine/blob/main/code/cuda/include/cpp_llm_engine/cuda/cuda_execution_context.h"><code>cuda_execution_context.h</code></a> | stream/handle/workspace 与 Tensor 级调度边界 |
| <a href="https://github.com/datawhalechina/cpp-llm-engine/blob/main/code/cuda/tests/cuda_execution_context_test.cu"><code>cuda_execution_context_test.cu</code></a> | Tensor、调度、复用、移动和错误 shape 测试 |
| <a href="https://github.com/datawhalechina/cpp-llm-engine/blob/main/code/cuda/include/cpp_llm_engine/cuda/cuda_quantized_matrix.h"><code>cuda_quantized_matrix.h</code></a> | packed Q4_K/Q6_K device 权重所有权 |
| <a href="https://github.com/datawhalechina/cpp-llm-engine/blob/main/code/cuda/include/cpp_llm_engine/cuda/quantized_dequantize_backend.h"><code>quantized_dequantize_backend.h</code></a> | CUDA block 反量化 backend 接口 |
| <a href="https://github.com/datawhalechina/cpp-llm-engine/blob/main/code/cuda/src/quantized_dequantize_backend.cu"><code>quantized_dequantize_backend.cu</code></a> | Q4_K/Q6_K 到 Float32 workspace 的 kernel |
| <a href="https://github.com/datawhalechina/cpp-llm-engine/blob/main/code/cuda/tests/cuda_quantized_linear_test.cu"><code>cuda_quantized_linear_test.cu</code></a> | CPU golden、非融合 Linear 和固定 workload 计时 |
| <a href="https://github.com/datawhalechina/cpp-llm-engine/blob/main/code/core/include/cpp_llm_engine/core/rope_contract.h"><code>rope_contract.h</code></a> | 连续布局、旋转维度、position offset 和频率契约 |
| <a href="https://github.com/datawhalechina/cpp-llm-engine/blob/main/code/cuda/include/cpp_llm_engine/cuda/rope_backend.h"><code>rope_backend.h</code></a> | CUDA RoPE backend 接口 |
| <a href="https://github.com/datawhalechina/cpp-llm-engine/blob/main/code/cuda/src/rope_backend.cu"><code>rope_backend.cu</code></a> | 相邻分量 pair kernel 与 launch 检查 |
| <a href="https://github.com/datawhalechina/cpp-llm-engine/blob/main/code/cuda/tests/cuda_rope_test.cu"><code>cuda_rope_test.cu</code></a> | CPU golden、局部旋转、错误参数和 event 计时 |
| <a href="https://github.com/datawhalechina/cpp-llm-engine/blob/main/code/cuda/include/cpp_llm_engine/cuda/cuda_kv_cache.h"><code>cuda_kv_cache.h</code></a> | 请求级连续 device K/V 所有权与 append 接口 |
| <a href="https://github.com/datawhalechina/cpp-llm-engine/blob/main/code/cuda/src/cuda_kv_cache.cpp"><code>cuda_kv_cache.cpp</code></a> | shape 校验、异步 D2D append、clear 和释放 |
| <a href="https://github.com/datawhalechina/cpp-llm-engine/blob/main/code/cuda/tests/cuda_kv_cache_test.cu"><code>cuda_kv_cache_test.cu</code></a> | 容量、偏移、复用、移动和请求隔离测试 |
| <a href="https://github.com/datawhalechina/cpp-llm-engine/blob/main/code/core/include/cpp_llm_engine/core/gqa_attention_contract.h"><code>gqa_attention_contract.h</code></a> | GQA 布局、head 映射、position 和 scale 契约 |
| <a href="https://github.com/datawhalechina/cpp-llm-engine/blob/main/code/cuda/include/cpp_llm_engine/cuda/gqa_attention_backend.h"><code>gqa_attention_backend.h</code></a> | CUDA GQA backend 接口 |
| <a href="https://github.com/datawhalechina/cpp-llm-engine/blob/main/code/cuda/src/gqa_attention_backend.cu"><code>gqa_attention_backend.cu</code></a> | score、稳定 softmax 和 value 三阶段 kernel |
| <a href="https://github.com/datawhalechina/cpp-llm-engine/blob/main/code/cuda/tests/cuda_gqa_attention_test.cu"><code>cuda_gqa_attention_test.cu</code></a> | prefill/decode golden、错误路径和 event 计时 |
| <a href="https://github.com/datawhalechina/cpp-llm-engine/blob/main/code/core/include/cpp_llm_engine/core/decoder_elementwise_contract.h"><code>decoder_elementwise_contract.h</code></a> | RMSNorm 与 SwiGLU 请求契约 |
| <a href="https://github.com/datawhalechina/cpp-llm-engine/blob/main/code/cuda/src/decoder_elementwise_backend.cu"><code>decoder_elementwise_backend.cu</code></a> | RMSNorm 归约与 SwiGLU kernel |
| <a href="https://github.com/datawhalechina/cpp-llm-engine/blob/main/code/cuda/tests/cuda_decoder_elementwise_test.cu"><code>cuda_decoder_elementwise_test.cu</code></a> | 257/2048 宽度 CPU golden |
| <a href="https://github.com/datawhalechina/cpp-llm-engine/blob/main/code/cuda/include/cpp_llm_engine/cuda/cuda_decoder_layer.h"><code>cuda_decoder_layer.h</code></a> | 单 token Decoder 配置、权重视图和中间 Tensor |
| <a href="https://github.com/datawhalechina/cpp-llm-engine/blob/main/code/cuda/src/cuda_decoder_layer.cpp"><code>cuda_decoder_layer.cpp</code></a> | 完整单层算子编排 |
| <a href="https://github.com/datawhalechina/cpp-llm-engine/blob/main/code/cuda/tests/cuda_decoder_layer_test.cu"><code>cuda_decoder_layer_test.cu</code></a> | 合成小维度逐阶段 golden |
| <a href="https://github.com/datawhalechina/cpp-llm-engine/blob/main/code/cuda/include/cpp_llm_engine/cuda/cuda_linear_weight.h"><code>cuda_linear_weight.h</code></a> | Float32 与量化 Linear 的统一只读视图 |
| <a href="https://github.com/datawhalechina/cpp-llm-engine/blob/main/code/cuda/scripts/run.sh"><code>scripts/run.sh</code></a> | CUDA 12.8 隔离构建和 CTest 入口 |

## 验证

```bash
CUDA_HOME=/usr/local/cuda-12.8 bash code/cuda/scripts/run.sh
```
