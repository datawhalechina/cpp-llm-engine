# chapter04_attention 代码导航

本章代码包含 Attention、FFN、DecoderLayer、KV Cache 和对应测试。

## 阅读顺序

1. 先看 `attention.cpp`
2. 再看 `feed_forward.cpp` 和 `decoder_layer.cpp`
3. 最后运行测试，核对各模块的输入、输出和错误路径

## 模型结构

| 文件 | 作用 |
| --- | --- |
| <a href="/_code/chapter04_attention/src/attention.cpp"><code>src/attention.cpp</code></a> | 多头注意力 |
| <a href="/_code/chapter04_attention/src/feed_forward.cpp"><code>src/feed_forward.cpp</code></a> | FFN |
| <a href="/_code/chapter04_attention/src/decoder_layer.cpp"><code>src/decoder_layer.cpp</code></a> | DecoderLayer 组合 |
| <a href="/_code/chapter04_attention/src/layer_norm.cpp"><code>src/layer_norm.cpp</code></a> | LayerNorm |
| <a href="/_code/chapter04_attention/src/linear_layer.cpp"><code>src/linear_layer.cpp</code></a> | Linear |
| <a href="/_code/chapter04_attention/src/tensor.cpp"><code>src/tensor.cpp</code></a> | 本章 Tensor |

## 验证与脚本

| 文件 | 作用 |
| --- | --- |
| <a href="/_code/chapter04_attention/examples/main_attention.cpp"><code>examples/main_attention.cpp</code></a> | 示例入口 |
| <a href="/_code/chapter04_attention/tests/chapter04_attention_test.cpp"><code>tests/chapter04_attention_test.cpp</code></a> | Attention、KV Cache 和 Decoder 结构测试 |
| <a href="/_code/chapter04_attention/tests/chapter02_tensor_attention_integration_test.cpp"><code>tests/chapter02_tensor_attention_integration_test.cpp</code></a> | Tensor View 基础组件与 3D/4D Attention 集成测试 |
| <a href="/_code/chapter04_attention/scripts/run.sh"><code>scripts/run.sh</code></a> | 运行脚本 |

## 对应文档

- [chapter04 导读](../chapter04_attention/intro.md)
