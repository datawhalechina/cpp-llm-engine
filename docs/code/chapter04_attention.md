# chapter04_attention 代码导航

本章代码包含 Attention、稠密/MoE FFN、DecoderLayer、KV Cache 和对应测试。

## 阅读顺序

1. 先看 `attention.cpp`
2. 再看 `feed_forward.cpp`、`moe_feed_forward.cpp` 和 `decoder_layer.cpp`
3. 最后运行测试，核对各模块的输入、输出和错误路径

## 模型结构

| 文件 | 作用 |
| --- | --- |
| <a href="https://github.com/datawhalechina/cpp-llm-engine/blob/main/code/chapter04_attention/src/attention.cpp"><code>src/attention.cpp</code></a> | 多头注意力 |
| <a href="https://github.com/datawhalechina/cpp-llm-engine/blob/main/code/chapter04_attention/src/feed_forward.cpp"><code>src/feed_forward.cpp</code></a> | FFN |
| <a href="https://github.com/datawhalechina/cpp-llm-engine/blob/main/code/chapter04_attention/src/moe_feed_forward.cpp"><code>src/moe_feed_forward.cpp</code></a> | Router、Top-K、专家与共享专家合并 |
| <a href="https://github.com/datawhalechina/cpp-llm-engine/blob/main/code/chapter04_attention/src/decoder_layer.cpp"><code>src/decoder_layer.cpp</code></a> | DecoderLayer 组合 |
| <a href="https://github.com/datawhalechina/cpp-llm-engine/blob/main/code/chapter04_attention/src/layer_norm.cpp"><code>src/layer_norm.cpp</code></a> | LayerNorm |
| <a href="https://github.com/datawhalechina/cpp-llm-engine/blob/main/code/chapter04_attention/src/linear_layer.cpp"><code>src/linear_layer.cpp</code></a> | Linear |
| <a href="https://github.com/datawhalechina/cpp-llm-engine/blob/main/code/chapter04_attention/src/tensor.cpp"><code>src/tensor.cpp</code></a> | 本章 Tensor |

## 验证与脚本

| 文件 | 作用 |
| --- | --- |
| <a href="https://github.com/datawhalechina/cpp-llm-engine/blob/main/code/chapter04_attention/examples/main_attention.cpp"><code>examples/main_attention.cpp</code></a> | 示例入口 |
| <a href="https://github.com/datawhalechina/cpp-llm-engine/blob/main/code/chapter04_attention/tests/chapter04_attention_test.cpp"><code>tests/chapter04_attention_test.cpp</code></a> | Attention、KV Cache 和 Decoder 结构测试 |
| <a href="https://github.com/datawhalechina/cpp-llm-engine/blob/main/code/chapter04_attention/tests/moe_feed_forward_test.cpp"><code>tests/moe_feed_forward_test.cpp</code></a> | MoE 路由与专家计算固定结果测试 |
| <a href="https://github.com/datawhalechina/cpp-llm-engine/blob/main/code/chapter04_attention/tests/chapter02_tensor_attention_integration_test.cpp"><code>tests/chapter02_tensor_attention_integration_test.cpp</code></a> | Tensor View 基础组件与 3D/4D Attention 集成测试 |
| <a href="https://github.com/datawhalechina/cpp-llm-engine/blob/main/code/chapter04_attention/scripts/run.sh"><code>scripts/run.sh</code></a> | 运行脚本 |
| <a href="https://github.com/datawhalechina/cpp-llm-engine/blob/main/code/chapter04_attention/scripts/run_qwen15_moe_ffn.sh"><code>scripts/run_qwen15_moe_ffn.sh</code></a> | 运行真实 Qwen1.5-MoE layer-0 FFN smoke |
| <a href="https://github.com/datawhalechina/cpp-llm-engine/blob/main/code/chapter04_attention/scripts/run_qwen15_moe_ffn_compare.sh"><code>scripts/run_qwen15_moe_ffn_compare.sh</code></a> | 使用 llama.cpp 中间张量核对 layer-0 Attention、FFN 与层输出 |
| <a href="https://github.com/datawhalechina/cpp-llm-engine/blob/main/code/chapter04_attention/scripts/run_qwen15_moe_prefill_compare.sh"><code>scripts/run_qwen15_moe_prefill_compare.sh</code></a> | 核对 layer-0 双词元 NeoX RoPE、因果 Attention、K/V 与层输出 |
| <a href="https://github.com/datawhalechina/cpp-llm-engine/blob/main/code/chapter04_attention/scripts/run_qwen15_moe_two_layer_compare.sh"><code>scripts/run_qwen15_moe_two_layer_compare.sh</code></a> | 核对双词元两层 hidden 交接、分层 K/V 和 layer-1 输出 |
| <a href="https://github.com/datawhalechina/cpp-llm-engine/blob/main/code/chapter04_attention/scripts/run_qwen15_moe_decoder_compare.sh"><code>scripts/run_qwen15_moe_decoder_compare.sh</code></a> | 核对双词元 24 层局部 Top-K、逐层数值、K/V、logits 和贪心词元 |

## 对应文档

- [chapter04 导读](../chapter04_attention/intro.md)
