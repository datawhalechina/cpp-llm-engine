# Chapter 04：Attention 与 Decoder 结构

本目录把 MatMul、Attention、LayerNorm、FFN 组合成可测试的 DecoderLayer，为 Chapter 06 generation runtime 提供结构输入。完整学习说明见 [`source/chapter04_attention`](../../source/chapter04_attention/intro.md)。

## 本章资产总览

| 资产 | 作用 |
| --- | --- |
| `include/` | Attention、DecoderLayer、backend 和结构契约适配 |
| `src/attention.cpp` | causal multi-head attention 及二维公共 View 入口 |
| `src/attention_backend.cpp` | CPU reference attention backend 适配 |
| `src/attention_kernel.cpp` | CPU causal attention kernel |
| `src/tensor_attention_adapter.cpp` | 3D/4D Q/K/V 与可写输出 View 到 Attention backend 的适配 |
| `src/kv_cache.cpp` | 固定容量 KV Cache append/clear |
| `src/layer_norm.cpp` | LayerNorm 及公共只读/可写 View 入口 |
| `src/feed_forward.cpp` | FFN、ReLU 及公共只读/可写 View 入口 |
| `src/decoder_layer.cpp` | 公共 View 上组合 Attention、残差和 FFN，并输出结构契约 |
| `src/linear_layer.cpp` | Decoder 内部线性层，以及公共只读/可写 View 入口 |
| `tests/chapter04_attention_test.cpp` | Attention、decode offset、KV Cache 和输出测试 |
| `tests/chapter02_tensor_attention_integration_test.cpp` | Chapter02 Tensor 进入基础算子、FFN 和 Attention 的跨章测试 |
| `examples/main_attention.cpp` | Decoder 结构演示 |
| `scripts/run.sh` | 隔离构建、CTest 和 demo 入口 |
| `scripts/run_real_structure.sh` | TinyLlama 首层九个权重的真实结构装配 | `real_tinyllama_first_layer_structure_test` |
| `code/core` | 共享 backend 和 operator 契约 |

## 运行与验证

从仓库根目录执行：

```bash
bash code/chapter04_attention/scripts/run.sh
```

重点观察：二维激活怎样通过公共只读/可写 View 进入 Linear，3D `[T,H,Dh]` 与 4D `[B,T,H,Dh]` 输入怎样进入同一 CPU Attention backend，以及非连续 View 为什么必须先显式连续化。

准备 TinyLlama 后，运行首层真实结构验收：

```bash
bash code/chapter04_attention/scripts/run_real_structure.sh
```

该入口验证九个权重的名称、shape、dtype 和 payload。量化权重进入数值 forward 后由 Chapter05 验收。
