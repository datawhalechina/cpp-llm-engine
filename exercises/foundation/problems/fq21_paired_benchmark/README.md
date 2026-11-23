# FQ21 Paired Benchmark

## 任务

实现一个最小配对性能实验：先预热，再交替执行参考实现与候选实现，记录每轮耗时，检查结果一致后计算中位数。实验只报告数据，不把“候选实现必须更快”写成正确性断言。

## 完成标准

- 预热样本不进入正式统计。
- 调用计数能区分预热次数与正式测量次数。
- 每轮交替 `baseline-first` 和 `candidate-first`，降低固定顺序偏差。
- 每次正式测量都先通过结果容差检查。
- 分别保留两条路径的原始样本并计算中位数。
- 零次迭代、负容差和结果不一致返回明确错误。

## 验证

复制并完成 starter 后，先运行自己的答案：

```bash
cp -r exercises/foundation/problems/fq21_paired_benchmark/starter /tmp/fq21-answer
bash scripts/check_foundation_answer.sh /tmp/fq21-answer/main.cpp
```

运行 `bash exercises/foundation/run_all.sh`，重点检查 `fq21_paired_benchmark` 的预热、交替顺序、正确性门禁和样本统计。

## 提示

- L0：用 `std::chrono::steady_clock` 测量单次调用。
- L1：偶数轮先运行参考实现，奇数轮先运行候选实现。
- L2：排序副本计算奇数和偶数样本的中位数，同时保留原始采样顺序用于复查。

## 主线映射

对应 [`Foundation 性能实验`](/foundation/13_performance_measurement)、[`Chapter09 性能证据`](/chapter09_performance/next/intro) 和 [`Chapter05 量化性能`](/chapter05_quantization/intro)。
