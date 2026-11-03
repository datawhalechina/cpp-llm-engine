# 06.6 从候选分数选择下一个词元

Sampler 把最后一个位置的 logits 转成下一个 token。仓库当前可执行实现是确定性的 `GreedySampler`；Temperature、Top-K 和 Top-p 用于定义后续扩展契约。

实现入口是 <a href="../_code/chapter06_generation/src/sampler.cpp"><code>sampler.cpp</code></a>，公式和数值约束见 [`F-SAMPLE-01 Greedy 与 Temperature`](../reference/formulas.md#f-sample-01)。

## Greedy 决策

Greedy 返回最大 logit 的 token ID，不需要先计算 softmax，因为 softmax 保持大小顺序。

```cpp
int sample_greedy(const Tensor& logits) {
    return argmax(logits);
}
```

模型负责产生分数，Sampler 负责做 token 决策。分离这两个对象后，固定 logits 可以独立测试采样，模型数值问题也不会和随机策略混在一起。

| 输入或状态 | 当前检查 |
| --- | --- |
| logits shape | 必须包含可采样的词表维度 |
| logits value | 最大值对应固定 token ID |
| 重复运行 | 相同 logits 得到相同结果 |
| Request 状态 | 采样后由 Runtime 追加 token 并检查 EOS/budget |

## 扩展策略

Temperature 在 softmax 前缩放 logits：`softmax(logits / temperature)`。较高 temperature 使分布更平滑，较低值使结果更集中；接近 0 时应切换到 Greedy 或采用明确的数值保护。

Top-K 保留分数最高的 K 个候选，Top-p 保留累计概率达到阈值的最小候选集。两者都需要重新归一化并从候选集中采样。

| 策略 | 候选集 | 随机性 | 需要新增的测试 |
| --- | --- | --- | --- |
| Greedy | 最大值一个 token | 无 | argmax、重复性、异常输入 |
| Temperature | 全词表 | 由温度控制 | 极值、数值稳定性 |
| Top-K | 固定 K 个 token | 有 | K 的范围、候选集 |
| Top-p | 累计概率集合 | 有 | 排序、阈值和重新归一化 |

请求级扩展应把参数放在独立 `SamplerConfig`，由应用层为请求创建 Sampler。`Request` 继续保存 token history 和生命周期，模型与 Scheduler 无需读取 temperature、K 或 p。

## 验证与衔接

```bash
bash code/chapter06_generation/scripts/run.sh
```

现有测试检查最大 logit 对应的 token ID以及重复运行结果。实现随机策略时，再增加固定随机种子、候选集、概率归一化和非法参数测试。下一页 [`06.7 端到端生成测试`](./07_end_to_end_generation.md) 会把 Request、Executor、Scheduler 和 Sampler 放入同一生成闭环。
