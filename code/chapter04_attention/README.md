# Chapter 04：Attention 与 Decoder 结构

本目录先用小尺寸 Float32 权重把 LayerNorm、Multi-Head Attention、ReLU FFN 和残差连接组成一个可测试的 DecoderLayer，再用真实结构脚本核对 TinyLlama 首层权重。原理和图解见 [Chapter04 导读](../../source/chapter04_attention/intro.md)。

## 代码架构

```text
输入 [tokens, model_dim]
  │
  ├─ LayerNorm → Q/K/V Linear → CPU Attention → Output Linear → 残差
  │
  └─ LayerNorm → Linear → ReLU → Linear ─────────────────────→ 残差
                                                                │
                                                                ▼
                                                      单层 Decoder 输出

每个请求的 DecoderLayerContext
  └─ prefill 写入多个 K/V → decode 追加一个 K/V → 成功后提交状态
```

这条教学路径用于看清单层数据流，不冒充 TinyLlama 的真实数学结构。真实模型使用 RMSNorm、RoPE、GQA、SwiGLU、量化 Linear 和模型级 KV 状态；对应权重由 Chapter01 加载，量化计算由 Chapter05 提供，完整多层执行位于 `code/model`，生成状态进入 Chapter06。

## 阅读顺序与资产

| 资产 | 作用 |
| --- | --- |
| `include/` | Attention、DecoderLayer、backend 和结构契约适配 |
| `src/attention.cpp` | causal multi-head attention 及二维公共 View 入口 |
| `src/attention_backend.cpp` | CPU reference attention backend 适配 |
| `src/attention_kernel.cpp` | CPU causal attention kernel |
| `src/tensor_attention_adapter.cpp` | 3D/4D Q/K/V 与可写输出 View 到 Attention backend 的适配 |
| `src/kv_cache.cpp` | 固定容量 KV Cache append/clear |
| `src/decoder_context.cpp` | 单请求、单层连续 K/V 的生命周期 |
| `src/layer_norm.cpp` | LayerNorm 及公共只读/可写 View 入口 |
| `src/feed_forward.cpp` | FFN、ReLU 及公共只读/可写 View 入口 |
| `src/decoder_layer.cpp` | 公共 View 上组合 Attention、残差和 FFN，并输出结构契约 |
| `src/linear_layer.cpp` | Decoder 内部线性层，以及公共只读/可写 View 入口 |
| `tests/chapter04_attention_test.cpp` | Attention、decode offset、KV Cache 和输出测试 |
| `tests/chapter02_tensor_attention_integration_test.cpp` | Chapter02 Tensor 进入基础算子、FFN 和 Attention 的跨章测试 |
| `examples/main_attention.cpp` | Decoder 结构演示 |
| `scripts/run.sh` | 隔离构建、CTest 和 demo 入口 |
| `scripts/run_real_structure.sh` | TinyLlama 首层九个权重的真实结构装配 |
| `code/core` | 共享 backend 和 operator 契约 |

先读 `attention_kernel.cpp`，单独理解遮罩和 Softmax；再读 `attention.cpp` 的四个投影和缓存入口；最后读 `decoder_layer.cpp` 与 `decoder_context.cpp`，观察双残差、prefill/decode 和成功后提交。连续 KV 只表达数值语义，分页管理属于 Chapter07。

## 运行与验证

从仓库根目录执行：

```bash
bash code/chapter04_attention/scripts/run.sh
```

重点观察：二维激活怎样通过公共只读/可写 View 进入 Linear，3D `[T,H,Dh]` 与 4D `[B,T,H,Dh]` 输入怎样进入同一 CPU Attention backend，以及单层 Context 怎样把缓存长度从 2 推进到 3。非连续 View 必须先显式连续化，缓存容量失败不能提交状态或输出。

准备 TinyLlama 后，运行首层真实结构验收：

```bash
bash code/chapter04_attention/scripts/run_real_structure.sh
```

该入口验证九个权重的名称、shape、dtype 和 payload。量化权重进入数值 forward 后由 Chapter05 验收。

## 扩展入口

替换 Attention 计算时实现 Core `AttentionBackend`，并保证 backend 生命周期长于 `MultiHeadAttention`。真实 Llama 结构不要继续堆入教学 `DecoderLayer`，应通过 Core `DecoderLayerContract` 与 `DecoderExecutionContext` 接入 `code/model` 或 CUDA 执行上下文。
