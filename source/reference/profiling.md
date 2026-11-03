# 推理引擎 Profiling

Benchmark 衡量性能变化，Profiling 定位时间消耗在 kernel、内存、调度、数据搬运还是并发等待。

## 1. Benchmark 之后看什么

benchmark 回答的是“快了多少、慢了多少”。  
profiling 回答的是“为什么会这样”。

定位性能变化时，需要继续判断改动落在：

- kernel
- 内存
- 调度
- 数据搬运
- 并发竞争

瓶颈位置决定应该优化 kernel、数据布局、调度还是并发控制。

## 2. benchmark 和 profiling 的分工

| 问题 | benchmark | profiling |
| --- | --- | --- |
| 目标 | 对照结果 | 解释原因 |
| 输出 | latency、throughput、prefill 时间 | 热点、调用路径、耗时分布 |
| 关注点 | 固定条件下的性能比较 | 瓶颈归因 |
| 作用 | 做决策 | 指导优化 |

这两者不能混在一起。  
benchmark 是证据，profiling 是证据说明书。

## 3. Profiling 里最先看什么

推理引擎里的热点通常出现在几类地方：

- 计算热点：kernel、matmul、attention、dequantize
- 内存热点：拷贝、分配、回收、碎片整理
- 调度热点：请求推进、batch 组装、队列争用
- 流程热点：prefill / decode 切换、cache 维护、输出回调

先分清热点属于哪一类，再决定下一步是改 kernel、改内存，还是改调度。

## 4. 常见的解释路径

一个实用的 profiling 顺序通常是：

1. 先确认 benchmark 的输入、硬件、运行方式是否固定
2. 再看耗时主要落在哪个阶段
3. 再把阶段拆成计算、内存、调度和流程四类
4. 最后再去看更细的函数、算子或 kernel

这个顺序的价值在于，先把大问题缩小到正确的层级，再决定要不要看更细的调用栈。

## 5. 常见误区

- 把 profiling 结果直接当 benchmark 结果
- 看到某个函数最慢，就立刻改那个函数
- 只看单次运行，不看复现条件
- 只看 kernel，不看调度和内存
- 把局部热点当成全局瓶颈

这些误区会让优化方向偏掉，最后得到“看起来更忙、但不一定更快”的结果。

## 6. 和主线章节的关系

| 章节 | 已经做了什么 | profiling 继续补什么 |
| --- | --- | --- |
| `chapter07_benchmark` | 正确性、benchmark、结果记录 | 为什么快、为什么慢 |
| `chapter08_profiling` | profiling、A/B 对比、优化闭环 | 把瓶颈解释变成可执行优化流程 |
| `chapter06_generation` | 生成 runtime 骨架 | runtime 内部哪个环节在吃时间 |
| `chapter05_quantization` | 表示和成本变化 | 误差和成本变化到底落在哪一层 |

profiling 是把 benchmark 的结论变成下一步优化动作的桥梁。

## 7. 如何使用 profiling 参考页

- 先用 `chapter07_benchmark` 确认结果可信
- 再读 [`chapter08_profiling`](../chapter08_profiling/intro.md) 建立 profiling 和 A/B 对比流程
- 最后回到这里找瓶颈归因的切入点
- 最后再决定是改 runtime、改内存，还是改后端

## 核心结论

Benchmark 衡量结果，Profiling 解释原因。
