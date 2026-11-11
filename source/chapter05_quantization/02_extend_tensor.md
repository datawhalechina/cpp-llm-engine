# 05.2 用 Tensor 表达教学量化数据

量化公式要进入代码，首先需要保存整数数据、shape 和缩放因子。本节使用 Chapter05 的最小二维 `Tensor` 对照 Float32 与 INT8 表示；它是量化教学对象，不替代 Chapter02 的通用 N 维 Tensor。

## 数据类型与存储

<a href="https://github.com/datawhalechina/cpp-llm-engine/blob/main/code/chapter05_quantization/include/chapter05_quantization/tensor.h"><code>tensor.h</code></a> 当前只声明两种数据类型：

```cpp
enum class DType {
    Float32,
    Int8,
};
```

Float32 和 INT8 分别保存在类型明确的共享数组中，`dtype` 决定哪个数组有效。INT8 Tensor 还保存一个标量 `scale`，`at()` 使用 `q * scale` 返回浮点观察值，`at_int8()` 则读取原始整数。

| 对象 | 保存内容 | 允许的观察 |
| --- | --- | --- |
| Float32 Tensor | 行列、`vector<float>` | `at(row, col)` |
| INT8 Tensor | 行列、`vector<int8_t>`、scale | `at_int8()`、`at()`、`dequantize()` |
| Chapter02 Tensor | N 维 shape、stride、offset、共享 storage | reshape、slice、连续化和公共 View |

量化规则没有塞进通用 storage。Chapter05 的对象用于讲清 dtype 与反量化；跨章执行则由公共只读/可写 View 把 Chapter02 激活交给量化 Linear。

## 构造、读取与失败条件

`Tensor::from_float()` 和 `Tensor::from_int8()` 检查 `rows * cols` 是否与输入元素数一致。空对象保持 `ready()==false`，错误 shape、越界索引和无效 scale 必须被测试覆盖，不能静默解释为另一种数据类型。

新增数据类型不等于新增量化格式。Q4_K/Q6_K 的 packed payload 具有独立 block 布局，由 `QuantizedMatrixView` 和对应解码器表达，而不是加入这个教学 `Tensor` 的元素访问接口。

## 验证与衔接

```bash
bash code/chapter05_quantization/scripts/run.sh
```

确认 Float32 旧路径仍通过、INT8 原始值和反量化值符合预期，并且非法输入返回明确失败。下一篇 [`05.3 接收量化权重`](./03_load_quantized_weights.md) 将区分 Chapter01 的文件解析结果与 Chapter05 的计算输入。
