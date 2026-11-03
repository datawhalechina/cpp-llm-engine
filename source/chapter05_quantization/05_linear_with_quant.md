# 05.5 在量化权重上执行线性计算

Decoder 上层只需要提交激活、权重格式和输出空间，不应因权重采用 Float32、Q4_K、Q6_K 或 Q8_0 而改变调用方式。本节用 reference 与 fused 两条路径证明：执行策略可以变化，Linear 的 shape 和数值契约保持不变。

## 从参考路径到融合线性层

reference 路径先还原完整浮点权重，再执行 `W * input`；fused 路径读取一个量化 block 后立即完成乘加，不创建完整权重数组。下面的算法镜像省略 backend 注册、SIMD 指令、CUDA workspace 和错误恢复，只比较两条路径决定结果的动作。

```python
def linear_reference(packed_weight, input_vector):
    float_weight = dequantize_all_blocks(packed_weight)
    return matrix_vector_product(float_weight, input_vector)


def linear_fused(packed_weight, input_vector):
    output = zeros(packed_weight.output_rows)
    for output_row in range(packed_weight.output_rows):
        for block in packed_weight.row_blocks(output_row):
            for column, weight in decode_block(block):
                output[output_row] += weight * input_vector[column]
    return output
```

| 算法动作 | C++ 对象或源码 | 验证重点 |
| --- | --- | --- |
| 建立浮点基线 | `Q4KLinearReference`、`Q6KLinearReference` | 输出 shape 与固定数值 |
| block 内直接乘加 | `q4k_kernel.cpp`、`q6k_kernel.cpp` | 不创建完整 float 权重数组 |
| 选择执行路径 | `QuantizedLinearPolicy` | 格式、backend 与硬件能力匹配 |
| 返回统一结果 | `compute_quantized_linear()`、`compute_quantized_linear_batch()` | reference、单行与 batch 容差一致 |

实现顺序固定为 `Scalar reference -> fused scalar -> SIMD fused -> CUDA dequantize + SGEMM baseline -> CUDA fused candidate`。每增加一条路径，先与上一层做数值回归，再进入 benchmark。

## 计算后端策略与统一接口

`Q4KLinearAvx2` 和 `Q6KLinearAvx2` 在读取 block 时直接用 AVX2/FMA 累加。CUDA 首版由 `CudaQuantizedMatrix` 保存 packed payload，kernel 解码到复用 workspace，再调用 cuBLAS SGEMM。Q8_0 当前只有 portable scalar fused；请求不存在的 AVX2 实现会返回明确 `Status`，不会静默回退。

```cpp
chapter05_quantization::QuantizedLinearPolicy policy{
    chapter05_quantization::QuantizedLinearBackendKind::Avx2Fused,
    chapter05_quantization::QuantizedLinearBackendKind::ScalarFused,
    chapter05_quantization::QuantizedLinearBackendKind::ScalarFused
};
auto status = chapter05_quantization::compute_quantized_linear(
    request, policy, registry);
```

| 路径 | 中间权重 buffer | 计算方式 | 当前用途 |
| --- | --- | --- | --- |
| Reference | 完整 float 数组 | 先反量化，再标量点积 | 格式与数值基线 |
| Scalar fused | 无 | 按 block 解码并标量累加 | 默认可移植路径 |
| AVX2/FMA fused | 无 | 分组解码并向量乘加 | 显式 CPU 性能实验 |
| CUDA baseline | Float32 workspace | CUDA 解码后调用 cuBLAS SGEMM | GPU correctness 与 fused A/B 基线 |

真实 Decoder 按 Attention、FFN 和 LM Head 分别保存 policy。Q4_K gate/up、Q6_K down、Q4_K QKV/out 复用同一 dispatcher，只改变对应作用域的策略。生产框架怎样把 quant method、硬件和 kernel 绑定，见 [`05.7 量化实现与成熟框架对照`](./07_tensorrt_gemma_vllm_comparison.md)。

下图从公共线性层请求开始，展示格式分派、CPU/CUDA 路径和共同的数值基线。并列后端是可替换实现，不表示按顺序全部执行。

![Q4_K 量化线性层的后端分派](../assets/diagrams/chapter05-q4k-backend.svg)

## Packed 权重与激活视图

`QuantizedMatrixView` 保存 packed payload 指针、字节数、权重行列和 Q4_K/Q6_K/Q8_0 格式；它不提供普通 Float32 Tensor 的索引语义。`compute_quantized_linear_views()` 让 Chapter02 的 Float32 激活通过公共只读/可写 View 进入现有 policy 和 backend：rank-1 表示单 token，rank-2 表示多个 token 行。

rank-2 会构造 `LinearBatchRequest`。Q4_K、Q6_K 和 Q8_0 的 scalar fused backend 只遍历一次 packed 权重，把同一个解码值分别累加到各输入行；这样得到的是 CPU 单线程批量 GEMV，而不是逐行重复解码权重。AVX2 backend 目前通过公共兼容路径逐行执行，尚未提供专用多行 SIMD kernel。

所有 batch 行必须连续且宽度相同，输出采用 `[batch, output_rows]`。适配器先写临时结果，整批成功后才提交到输出 View；非连续激活必须由调用方显式 `contiguous()`，packed 权重也必须严格匹配格式对应的 block 字节数。

## 验证与衔接

```bash
bash code/chapter05_quantization/scripts/run.sh
```

测试比较 reference、scalar fused 和可用的 AVX2 路径，检查输出 shape、数值容差、非法 backend 请求和真实 TinyLlama contiguous Q4_K row。跨章测试还验证 Q4_K、Q6_K、Q8_0 的多行 scalar batch、非连续输入拒绝和失败不覆盖输出。真实 Decoder 再用两个不同长度 prompt 对照顺序与 batch logits。下一节 [`05.6 精度验证`](./06_accuracy_validation.md) 将局部误差扩展到中间 Tensor 与最终输出。
