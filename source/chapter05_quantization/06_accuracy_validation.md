# 05.6 验证量化误差与最终输出

量化路径不能只证明“可以运行”，还要说明它与浮点基线相差多少、误差从哪里开始出现。验证使用相同输入比较 reference、fused backend 和真实 Decoder 输出；性能收益留给 Chapter09，不与正确性阈值混在一起。

## 局部误差与端到端输出

局部验证选择量化路径与浮点路径共享接口的位置，例如 Linear 输出、Attention 输出或 FFN 输出。它能定位第一个偏离点；最终 token 或 logits 则判断误差传播后是否改变模型行为。

```python
reference_output = float_path(input_tensor)
quantized_output = quantized_path(input_tensor)

max_error = max_abs(reference_output - quantized_output)
similarity = cosine_similarity(reference_output, quantized_output)
assert max_error <= format_tolerance
assert similarity >= similarity_threshold
```

这段比较省略模型加载、计时和硬件选择，只定义相同输入下的结果关系。容差必须随格式、计算阶段和输出尺度记录，不能用一个数字覆盖所有 Tensor。

| 比较层级 | 输入必须固定 | 主要证据 | 失败时定位 |
| --- | --- | --- | --- |
| block 解码 | packed payload、格式参数 | 逐元素最大误差 | layout、scale、bit unpack |
| Linear | 激活、权重、shape | max/mean error、cosine | 解码或累加顺序 |
| Decoder 中间量 | prompt、模型、backend policy | Attention/FFN 输出差值 | 首个偏离层和 projection |
| 最终生成 | tokenizer、prompt、Sampler | logits 数值指标、greedy token 序列 | 误差传播是否改变决策 |

## 仓库中的验证证据

Q4_K CPU reference 使用真实 TinyLlama `token_embd.weight` 与 llama.cpp reference 交叉验证；Q6_K 使用 `blk.0.ffn_down.weight` 做同类验证。Q8_0 使用合成 block 和真实 SmolVLM 路径检查 decode-and-dot、embedding-prefill 与完整 Decoder 输出。

| 验证对象 | 当前证据 | 通过标准 |
| --- | --- | --- |
| INT8 教学格式 | 固定整数、scale 和预期浮点值 | `q * scale` 与预期一致 |
| Q4_K/Q6_K 解码 | 真实 GGUF payload 对照 reference | 元素数、layout 和数值一致 |
| fused CPU Linear | reference、scalar、AVX2 同输入 | 输出落在格式容差内 |
| 真实 Decoder policy | 固定 TinyLlama prompt，同轮 llama.cpp reference | logits 满足误差阈值、greedy 序列一致、双 Context 相互隔离 |
| Q8_0 多模态路径 | token-prefill 与 embedding-prefill | logits/top-1 一致 |

性能 A/B 的硬件、编译器、warmup 和多轮统计集中在[实验与验证资产索引](../reference/experiment_assets.md)，并由 Chapter09 统一解释。不能把不同机器、模型、shape 或 backend 的耗时直接相减。

## 验证与衔接

```bash
bash code/chapter05_quantization/scripts/run.sh
```

测试应同时覆盖正确输入、非法 payload/shape、reference 与 fused 差值以及不支持 backend 的明确错误。正确性通过后先阅读 [`05.7 量化实现与成熟框架对照`](./07_tensorrt_gemma_vllm_comparison.md)，性能比较统一进入 Chapter09 的证据链。
