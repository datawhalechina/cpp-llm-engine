# 09.7 用单变量 A/B 验证优化

A/B 实验只改变一个待验证因素。baseline 与 experiment 必须通过相同正确性闸门，并使用相同 workload、模型、后端、构建类型、编译器和硬件身份。

当前比较器还要求两侧使用相同的预热次数和测量次数。策略必须在实验前选择延迟或吞吐作为主指标；主指标没有改善时，即使保护指标没有越界，也不会输出 `KEEP`。真实服务入口默认预热 1 轮、测量 3 轮。

## 实验步骤

| 步骤 | 操作 | 产物 |
| --- | --- | --- |
| 1 | 写出优化假设和主指标 | 实验说明 |
| 2 | 运行 baseline 并保存原始样本 | baseline 测量记录 |
| 3 | 只改变一个实现或参数 | 可审查的差异 |
| 4 | 运行 experiment 并保存原始样本 | experiment 测量记录 |
| 5 | 检查可比性和正确性 | 比较资格 |
| 6 | 计算变化并应用策略 | 独立对比记录 |

## 常见的无效比较

对比器应先返回 `NOT_COMPARABLE` 或正确性失败，再计算性能差异。不能用零值或默认值伪造一个可比较结论。

| 情况 | 为什么无效 |
| --- | --- |
| 同时改变算法、线程数和编译选项 | 无法判断哪个变化产生效果 |
| baseline 冷启动、experiment 已预热 | 缓存状态不同 |
| 只保留最快的一次 | 选择偏差掩盖波动 |
| 输出错误但速度更快 | 计算任务已经改变 |
| 不同输入长度共用 workload ID | Prefill/Decode 比例不可比 |

## 用配对顺序和矩阵检查结论

同一进程中先运行哪条路径，可能改变缓存、温度和设备时钟。教程的 CUDA 自适应 GEMV 与 CPU MoE 实验分别采用 `baseline-first` 和 `experiment-first` 两轮顺序，把同一工作负载下的样本配对保存。两轮方向不一致时，结论应标记为证据不足，而不是挑选有利的一轮。

单点 A/B 只回答一个工作负载。候选实现通过后，再改变 batch、上下文长度或 Decode 步数形成适用范围矩阵；每个单元仍执行正确性检查和反向顺序。矩阵中的控制组若也出现大幅波动，应先检查环境和测量流程，再解释实现差异。

## 验证比较器

<a href="https://github.com/datawhalechina/cpp-llm-engine/blob/main/code/chapter09_performance/src/comparison.cpp"><code>compare()</code></a> 先检查正确性、环境、workload、预热和测量轮数，再计算主指标与保护指标。执行：

```bash
bash code/chapter09_performance/scripts/run.sh
```

`chapter09_comparison_test` 应覆盖 `KEEP`、`REJECT`、`NOT_COMPARABLE` 和 `INVALID`，并证明主指标未改善时不会输出 `KEEP`。有真实模型时，`bash code/chapter09_performance/scripts/run_real_evidence.sh ab` 还会生成 `real-service-ab-decision.json`，其中保存 baseline、experiment、workload、阈值、变化和结论。`09.8` 把这四种结果转换为保留、回退或重测动作。
