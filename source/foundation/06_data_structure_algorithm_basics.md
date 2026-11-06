# F06 数据结构与算法基础

推理引擎的数据结构选择由访问方式决定：数值计算关注连续布局，请求管理关注顺序和状态，权重加载关注名称查找。先建立 CPU reference，再考虑优化。

跨章节的具体数据结构、算法、代码入口和验证产物见[数据结构与算法资产表](../reference/data_structures_algorithms.md)。

## 按访问方式选择数据结构

| 访问方式 | 首选结构 | 推理引擎例子 | 关注点 |
| --- | --- | --- | --- |
| 连续数值访问 | `vector` / buffer | Tensor payload | locality、stride |
| 名称查找 | `unordered_map` | 权重、metadata | key、缺失处理 |
| FIFO 顺序 | `deque` / queue | Request queue | 顺序、容量 |
| 固定小集合 | `array` | head、shape、维度 | 栈上布局 |

## 布局与复杂度

| 任务 | CPU baseline | 后续优化方向 |
| --- | --- | --- |
| MatMul | 清晰的 `O(MNK)` 循环 | SIMD、分块、BLAS |
| Tensor view | offset + stride | 连续访问和别名分析 |
| Request queue | 线性 FIFO | backpressure、批处理 |
| 权重查找 | 一次 map 查找 | 预解析、紧凑索引 |

复杂度只是第一层判断；真实推理还要看 cache、内存带宽、分支和 kernel launch。

## 最小示例

```cpp
for (std::size_t m = 0; m < M; ++m) {
  for (std::size_t n = 0; n < N; ++n) {
    float value = 0.0f;
    for (std::size_t k = 0; k < K; ++k)
      value += a[m * K + k] * b[k * N + n];
    c[m * N + n] = value;
  }
}
```

这段代码优先用于建立正确性证据，之后才能与 cuBLAS、SIMD 或 fused kernel 对比。

## 验证与练习

验证时说明数据为什么连续、查找为什么按名称、算法为什么先选可读 baseline。对应练习见 [Foundation 正文与练习映射](./exercise_map.md)，完成 `FQ03` 和 `FQ06`。
