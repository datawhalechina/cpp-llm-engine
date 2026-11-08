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

教学路径用小尺寸算子呈现单层数据流；真实模型路径使用 RMSNorm、RoPE、GQA、SwiGLU、量化 Linear 和模型级 KV 状态。Chapter01 加载对应权重，Chapter05 提供量化计算，`code/model` 执行完整多层 Decoder，Chapter06 管理生成请求状态。

## 阅读顺序与资产

| 顺序 | 文件 | 阅读重点 | 对应证据 |
| --- | --- | --- | --- |
| 1 | `src/attention_kernel.cpp`、`src/attention_backend.cpp` | 因果遮罩、稳定 Softmax 与 CPU backend 分派 | `chapter04_attention_test` 的 Attention kernel 用例 |
| 2 | `src/tensor_attention_adapter.cpp` | 3D/4D Q/K/V 的批次偏移、连续布局和失败原子性 | `chapter02_tensor_attention_integration_test`，integration |
| 3 | `src/attention.cpp`、`src/kv_cache.cpp` | 四个投影、位置偏移、候选 K/V 追加和成功提交 | `chapter04_attention_test` 的连续 KV 用例 |
| 4 | `src/layer_norm.cpp`、`src/feed_forward.cpp`、`src/linear_layer.cpp` | 按词元归一化、ReLU FFN 和公共二维 View | 两个测试 target 的组件与跨章用例 |
| 5 | `src/decoder_layer.cpp`、`src/decoder_context.cpp` | 双残差、prefill/decode、请求隔离和失败不提交 | `chapter04_attention_test`，component |
| 6 | `examples/main_attention.cpp` | 用 `2x4` 输入观察单层输出 | `chapter04_attention_demo` |
| 7 | `scripts/run_real_structure.sh` | 调用 Chapter01 核对 TinyLlama 首层九个权重 | `real_tinyllama_first_layer_structure_test`，real_model |

连续 K/V 在本章表达数值和请求状态语义；Chapter07 将相同逻辑位置映射到分页存储。

## 运行与验证

从仓库根目录执行：

```bash
bash code/chapter04_attention/scripts/run.sh
```

演示程序应输出 `ready: true`、`output shape: 2x4`，并打印位置 `(0,0)` 和 `(1,3)` 的有限浮点结果。CTest 随后报告三个 target 通过：Chapter04 component 测试覆盖 Attention、连续 KV、双残差和请求 Context；integration 测试覆盖 Chapter02 Tensor 到二维算子及 3D/4D Attention 的公开接口交接；Core unit 测试核对共享契约。

重点观察二维激活怎样通过公共只读/可写 View 进入 Linear，3D `[T,H,Dh]` 与 4D `[B,T,H,Dh]` 输入怎样进入同一 CPU Attention backend，以及单层 Context 怎样把缓存长度从 2 推进到 3。非连续 View 先显式连续化；缓存容量或输入缓冲区校验失败时，状态和输出保持不变。

准备 TinyLlama 后，运行首层真实结构验收：

```bash
bash code/chapter04_attention/scripts/run_real_structure.sh
```

该入口验证九个权重的名称、shape、dtype 和 payload。量化权重进入数值 forward 后由 Chapter05 验收。

## 扩展入口

GPU 路径通过 Core `AttentionBackend` 替换计算实现，backend 生命周期长于 `MultiHeadAttention`。真实 Llama 结构通过 Core `DecoderLayerContract` 与 `DecoderExecutionContext` 接入 `code/model` 或 CUDA 执行上下文；多模态路径在统一 embedding 序列处进入同一语言 Decoder。
