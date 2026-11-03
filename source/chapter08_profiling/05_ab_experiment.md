# 08.5 设计 A/B 对比实验

A/B 实验固定 workload ID、模型、输入和环境，每次只改变一个实现变量。baseline 与 experiment 可以登记为不同实验资产，但必须共享同一 workload ID；正确性、状态和资源检查先通过，性能数据才有比较意义。

## 固定实验契约

一次实验按以下顺序执行：

1. 写明假设、唯一变量和保留阈值。
2. 固定模型、数据类型、prompt、输出长度、batch、硬件和构建类型。
3. 对 baseline 与 experiment 运行相同的 golden、状态和资源检查。
4. 使用相同 warmup、迭代次数和统计方法采集结果。
5. 根据预设阈值作出 `KEEP`、`REJECT` 或 `SPLIT` 决定。

| 必填信息 | Baseline 与 Experiment 必须保持什么关系 |
| --- | --- |
| 环境 | CPU/GPU、编译器、构建类型和关键能力一致 |
| Workload ID | 模型、输入、输出、请求数和测量边界一致；被研究的执行方式可以不同 |
| Correctness | 两组都通过，否则停止性能比较 |
| 性能 | 同时保存逐轮值、汇总值和波动范围 |
| 唯一变量 | 只切换目标 backend、kernel 或调度策略 |

## 保存结果并作出决定

长期结果使用 JSON 或固定字段文本，至少记录 experiment ID、workload ID、环境、correctness、阶段耗时、吞吐和逐轮样本。真实服务回归中 baseline 是 `EXP-CPU-02`、复测是 `EXP-CPU-03`，但两者共享 `WL-CPU-REAL-SERVICE-V1`；比较器必须在 workload ID 或关键环境不一致时拒绝计算 delta。

第一次实验可以运行 `bash code/chapter07_benchmark/scripts/run.sh` 登记基线。真实实验的模型、硬件、原始结果、复现脚本和结论范围集中在[实验与验证资产索引](../reference/experiment_assets.md)。

| 决定 | 使用条件 |
| --- | --- |
| `KEEP` | 正确性通过，目标指标达到阈值，资源与复杂度代价可接受 |
| `REJECT` | 正确性失败或目标指标稳定回退 |
| `SPLIT` | 组合发生变化，但证据不足以定位单个变量 |

## 验证与衔接

通过标准是能在实验开始前写出唯一变量和回滚条件，并能解释为什么 correctness 失败后不能继续比较性能。下一节 [`08.6 优化闭环`](./06_optimization_loop.md) 将 A/B 决定接回“定位、修改、验证、保留或回滚”的完整流程。
