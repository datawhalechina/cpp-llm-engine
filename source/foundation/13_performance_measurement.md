# F13 性能测量基础

性能结论需要回答“在什么环境、什么输入、比较哪两条路径、结果是否仍然正确”。一次耗时只能说明一次运行发生了什么，不能直接证明优化有效。

## 先固定实验边界

| 项目 | 需要记录的内容 |
| --- | --- |
| 环境 | CPU/GPU、编译器、构建类型、线程数或 CUDA 版本 |
| 负载 | 模型、shape、batch、context、decode step |
| 路径 | 参考实现与候选实现的明确选择参数 |
| 正确性 | 相同输入下的误差、词元或资源状态 |
| 样本 | 预热次数、正式次数、原始耗时和统计量 |

Debug 构建适合定位错误，Release 构建适合性能比较。两条路径应在同一进程配置和尽量相近的系统状态下运行。

## 预热、配对和中位数

预热让延迟初始化、缓存填充和首次分配退出正式样本。正式实验可以交替执行：

```text
第 1 轮: baseline -> candidate
第 2 轮: candidate -> baseline
第 3 轮: baseline -> candidate
```

交替顺序降低固定先后顺序带来的偏差。中位数描述典型样本，最小值反映理想状态，最大值帮助发现抖动；报告中仍应保留原始样本。

## 正确性门禁先于速度

每个正式样本先确认输出或状态满足容差，再进入性能统计。若候选路径更快但结果错误、资源未释放或只处理了更少工作，该样本不能作为优化证据。性能测试本身也不应断言候选路径在所有机器上必然更快。

Profiler 用于解释时间花在哪里，Benchmark 用于比较端到端结果。先用 Benchmark 发现差异，再用 Nsight Systems、Nsight Compute 或 CPU Profiler 定位原因。

## 验证与练习

完成 [`FQ21 Paired Benchmark`](https://github.com/datawhalechina/cpp-llm-engine/blob/main/exercises/foundation/problems/fq21_paired_benchmark/README.md)，建立预热、交替顺序、正确性门禁和中位数统计。真实模型实验、证据文件与决策规则由 Chapter09 继续展开。
