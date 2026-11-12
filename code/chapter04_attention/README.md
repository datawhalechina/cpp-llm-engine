# Chapter 04：Attention 与 Decoder 结构

本目录先用小尺寸 Float32 权重把 LayerNorm、Multi-Head Attention、FFN 和残差连接组成一个可测试的 DecoderLayer，再用真实结构脚本核对模型权重。稠密 FFN 使用 ReLU 建立基础数据流，教学型 MoE FFN 额外展示 Router、稳定 Top-K、专家 SwiGLU 与共享专家合并。原理和图解见 [Chapter04 导读](../../source/chapter04_attention/intro.md)。

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

MoE FFN [tokens, model_dim]
  ├─ Router → Top-K → 选中专家 SwiGLU ─┐
  └─ 共享专家门控 → 共享专家 SwiGLU ──┴→ [tokens, model_dim]

MLA Attention
  ├─ 持久缓存：低秩 K/V + 共享 RoPE Key
  ├─ CPU reference：临时展开各头 K/V
  └─ 因果 Softmax → 各头 Value 输出
```

教学路径用小尺寸算子呈现单层数据流；真实模型路径使用 RMSNorm、RoPE、GQA、SwiGLU、量化 Linear 和模型级 KV 状态。Chapter01 加载对应权重，Chapter05 提供量化计算，`code/model` 执行完整多层 Decoder，Chapter06 管理生成请求状态。

## 阅读顺序与资产

| 顺序 | 文件 | 阅读重点 | 对应证据 |
| --- | --- | --- | --- |
| 1 | `src/attention_kernel.cpp`、`src/attention_backend.cpp` | 三段式 golden、分块在线 Softmax、因果遮罩与 CPU backend 分派 | `chapter04_attention_test`、`chapter04_online_softmax_test` |
| 1A | `src/mla_attention.cpp` | DeepSeek MLA 低秩缓存、各头临时展开和共享 RoPE Key | `chapter04_mla_attention_test` 与显式展开 golden |
| 2 | `src/tensor_attention_adapter.cpp` | 3D/4D Q/K/V 的批次偏移、连续布局和失败原子性 | `chapter02_tensor_attention_integration_test`，integration |
| 3 | `src/attention.cpp`、`src/kv_cache.cpp` | 四个投影、位置偏移、候选 K/V 追加和成功提交 | `chapter04_attention_test` 的连续 KV 用例 |
| 4 | `src/layer_norm.cpp`、`src/feed_forward.cpp`、`src/linear_layer.cpp` | 按词元归一化、ReLU FFN 和公共二维 View | 组件与跨章用例 |
| 5 | `src/moe_feed_forward.cpp`、`../model/src/real_moe_feed_forward.cpp` | Router、稳定 Top-K、按专家分发、选中专家与共享专家合并 | 教学固定 golden、分组/逐词元等价测试与真实 FFN smoke |
| 6 | `src/decoder_layer.cpp`、`src/decoder_context.cpp` | 双残差、prefill/decode、请求隔离和失败不提交 | `chapter04_attention_test`，component |
| 7 | `examples/main_attention.cpp` | 用 `2x4` 输入观察单层输出 | `chapter04_attention_demo` |
| 8 | `scripts/run_real_structure.sh` | 调用 Chapter01 核对 TinyLlama 首层九个权重 | `real_tinyllama_first_layer_structure_test`，real_model |
| 9 | `../model/src/real_qwen2_moe_layer.cpp` | NeoX RoPE、因果 Attention、连续 K/V 与真实 MoE 的单层执行 | llama.cpp 单层、两层与 24 层中间张量对照 |

连续 K/V 在本章表达数值和请求状态语义；Chapter07 将相同逻辑位置映射到分页存储。

DeepSeek-V2-Lite 当前包含四层证据：`src/mla_attention.cpp` 用小尺寸 Float32 数据核对低秩缓存语义；`code/model/src/real_deepseek2_dense_layer.cpp` 连接 CPU 归一化、YaRN partial RoPE、压缩历史 Attention 和双残差；`code/cuda/src/mla_attention_backend.cu` 保持同一压缩 Cache 契约并通过真实双词元稠密首层对照；`code/model/src/real_moe_feed_forward.cpp` 连接 64 选 6 路由、Q2_K/IQ4_NL 专家和共享专家。

完整 GGUF 可用时，先捕获 llama.cpp 首层逐节点参考：

```bash
bash code/chapter04_attention/scripts/run_deepseek_v2_lite_layer_capture.sh
```

该入口使用两个词元，固定检查每个词元 `2048` 维输入、`3072` 维 Query、`576` 维 K/V-A 输出、`512` 维归一化 latent，以及 Attention、FFN 和层输出。它提供真实模型 golden；`chapter04_mla_attention_test` 负责用小尺寸数据解释低秩缓存与显式展开为什么等价。

`real_deepseek2_dense_layer_test` 使用零投影固定 Context 增长、稠密/MoE 分支和失败原子性。真实捕获入口先逐节点比较第 0 层，再让第 1 层继续读取教程第 0 层输出；两层输出余弦相似度分别为 `0.999883` 和 `0.999859`。同一入口随后运行 1 个稠密层、26 个 MoE 层、最终归一化和 LM Head，双词元 logits 与 llama.cpp 的余弦相似度为 `0.999616`，贪心词元一致。

## 运行与验证

从仓库根目录执行：

```bash
bash code/chapter04_attention/scripts/run.sh
```

演示程序应输出 `ready: true`、`output shape: 2x4`，并打印位置 `(0,0)` 和 `(1,3)` 的有限浮点结果。CTest 随后报告六个测试通过：Chapter04 component 测试覆盖 Attention、连续 KV、双残差和请求 Context；MoE 测试覆盖稳定 Top-K、路由权重、专家与共享专家合并；在线 Softmax 测试对照三段式 golden；MLA 测试对照压缩缓存与显式展开路径；integration 测试覆盖 Chapter02 Tensor 的公开接口交接；Core unit 测试核对共享契约。

重点观察二维激活怎样通过公共只读/可写 View 进入 Linear，3D `[T,H,Dh]` 与 4D `[B,T,H,Dh]` 输入怎样进入同一 CPU Attention backend，以及单层 Context 怎样把缓存长度从 2 推进到 3。非连续 View 先显式连续化；缓存容量或输入缓冲区校验失败时，状态和输出保持不变。

准备 TinyLlama 后，运行首层真实结构验收：

```bash
bash code/chapter04_attention/scripts/run_real_structure.sh
```

该入口验证九个权重的名称、shape、dtype 和 payload。量化权重进入数值 forward 后由 Chapter05 验收。

准备 Qwen1.5-MoE 后，可执行真实 layer-0 FFN：

```bash
bash code/chapter04_attention/scripts/run_qwen15_moe_ffn.sh
bash code/chapter04_attention/scripts/run_qwen15_moe_ffn_compare.sh
bash code/chapter04_attention/scripts/run_qwen15_moe_prefill_compare.sh
bash code/chapter04_attention/scripts/run_qwen15_moe_two_layer_compare.sh
bash code/chapter04_attention/scripts/run_qwen15_moe_decoder_compare.sh
```

前四个入口依次验证真实 FFN、单词元层输出、双词元因果 Attention 和两层 hidden/K/V 交接。最后一个入口顺序加载 24 层权重，同时执行两种比较：局部比较把 llama.cpp 上一层输出交给当前层，核对 Top-K 与单层数值；累计比较把教程自己的输出继续传给下一层，观察量化误差经过离散路由后怎样放大。固定模型的局部 Top-K 全部一致，最大局部 MAE 为 `0.0418066`；累计路径最终 hidden MAE 为 `0.136931`、余弦相似度为 `0.997824`。最终 RMSNorm 和 Q6_K LM Head 继续生成词表 logits，贪心词元 `151643` 与 llama.cpp 一致。

## 扩展入口

教学 CPU 路径通过 Core `AttentionBackend` 注入计算实现，backend 生命周期长于 `MultiHeadAttention`。CUDA GQA 路径复用 Core 请求中的形状语义，并通过带 CUDA Stream 的独立接口进入设备执行上下文。真实模型结构通过 Core `DecoderLayerContract` 与 `DecoderExecutionContext` 接入 `code/model` 或 CUDA 执行上下文；多模态路径在统一 embedding 序列处进入同一语言 Decoder。
