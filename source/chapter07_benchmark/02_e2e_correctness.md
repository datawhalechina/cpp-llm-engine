# 07.2 分层验证端到端正确性

单个算子通过测试，不代表完整请求一定正确。本节沿“教学 Runtime、真实模型执行、服务请求”三条链路检查输入、状态和输出，并明确仓库当前没有提交 PyTorch golden logits 文件。

## 三类端到端证据

| 证据 | 固定输入 | 当前比较对象 | 能证明什么 |
| --- | --- | --- | --- |
| 教学 Runtime | 小词表、固定 prompt 和生成上限 | 请求完成数、生成数、缓存长度 | Chapter06 控制流和请求隔离成立 |
| 真实 TinyLlama | 固定 GGUF、prompt 和 greedy 策略 | 逐层数值指标、最终候选分数和固定词元 | 真实加载、22 层 Decoder 与量化执行能够接通 |
| 真实 ServiceCore | 固定双请求 | 串行/并发结果、Context 隔离和终态释放 | 服务包装没有改变模型结果和请求所有权 |

Chapter07 的 `run_e2e_checks()` 只检查固定阶段样例的结构，`chapter07_multi_request_benchmark` 检查教学 Runtime 行为。真实模型正确性由 `scripts/run_real_decoder_learning.sh` 提供，不能用前两者代替。

## 为比较对象选择指标

比较离散输出时，固定贪心词元是否一致是最直接的门槛；比较中间浮点向量时，可同时查看余弦相似度、最大绝对误差和平均绝对误差。阈值必须由数据类型、算子和参考路径共同确定，不能给所有阶段套用同一个数值。

真实模型脚本使用当前仓库固定的逐层检查和贪心词元作为回归证据。若以后加入 Python 或 PyTorch 参考，应把模型版本、tokenizer、输入 token、比较张量、生成脚本和阈值一起提交，不能只保存一组脱离来源的 logits。

正确性检查顺序如下：

1. 确认模型文件、tokenizer 和输入 token 一致。
2. 确认比较张量的 shape、数据类型和阶段一致。
3. 先检查非有限值和明显的 shape 错误，再计算误差指标。
4. 最后检查生成词元、请求终态和缓存释放。

## 验证与衔接

默认链路执行：

```bash
bash code/chapter07_benchmark/scripts/run.sh
```

有 TinyLlama GGUF 时再执行：

```bash
bash scripts/run_real_decoder_learning.sh
```

只有对应链路的正确性证据通过，性能数字才可进入比较。下一篇 [`07.3 性能指标`](./03_benchmark_metrics.md) 将固定每个指标的测量区间和单位。
