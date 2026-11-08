# 04.5 接入首层权重

Chapter01 已经把 TinyLlama 的模型配置、张量映射和 Llama 权重布局整理成可读取对象。本节沿这些对象追踪 `blk.0` 的九个权重怎样进入 RMSNorm、注意力、FFN 和残差计算；结构检查通过 Chapter04 脚本调用 Chapter01 的加载 target，不在 Chapter04 复制一套加载实现。

## Chapter01 交付的模型材料

首层计算不再解析 GGUF 字段或拼接文件偏移。加载章提供的对象分别回答配置、位置和数据三个问题：

| Chapter01 产物 | Chapter04 使用方式 |
| --- | --- |
| `ModelConfig` / `ModelMetadata` | 读取隐藏维度、注意力头数、RoPE 和 RMSNorm 参数 |
| `LlamaTensorLayout` | 找到 `blk.0` 对应的九个张量名称 |
| 张量映射 / 权重数据加载器 | 取得权重形状、数据类型和原始数据 |
| `RealLlamaLayerWeights` | 向首层计算提供已绑定字段的权重集合 |

真实权重的读取入口位于 <a href="../_code/chapter01_tensor_loader/src/real_layer_weights.cpp"><code>real_layer_weights.cpp</code></a>。本章脚本调用 <a href="../_code/chapter01_tensor_loader/examples/gguf_first_layer_inspect.cpp"><code>gguf_first_layer_inspect.cpp</code></a> 核对结构；接入 Chapter05 量化计算后的首层数值入口是 <a href="../_code/chapter01_tensor_loader/examples/gguf_first_layer_forward.cpp"><code>gguf_first_layer_forward.cpp</code></a>。

## 九个权重怎样进入解码器块

TinyLlama 使用 `blk.0.*` 命名。`RealLlamaLayerWeights` 将稳定字段连接到具体计算位置：

| 权重字段 | GGUF 张量 | 计算位置 |
| --- | --- | --- |
| `attn_norm` | `blk.0.attn_norm.weight` | 注意力前 RMSNorm |
| `attn_q` / `attn_k` / `attn_v` | `blk.0.attn_q/k/v.weight` | Q、K、V 投影 |
| `attn_output` | `blk.0.attn_output.weight` | 注意力输出投影 |
| `ffn_norm` | `blk.0.ffn_norm.weight` | FFN 前 RMSNorm |
| `ffn_gate` / `ffn_up` | `blk.0.ffn_gate/up.weight` | SwiGLU 两条输入投影 |
| `ffn_down` | `blk.0.ffn_down.weight` | FFN 输出投影 |

一次首层前向按以下顺序消费这些权重：

```text
token embedding
  -> RMSNorm
  -> Q/K/V projection
  -> RoPE + GQA attention
  -> output projection + residual
  -> RMSNorm
  -> gate/up projection + SwiGLU
  -> down projection + residual
  -> layer-0 hidden state
```

Q4_K、Q6_K 和 Q8_0 投影由 Chapter05 的量化线性计算后端执行；Chapter04 只决定这些投影在解码器块中的顺序和数据流。

## 验证与衔接

运行本章真实结构测试：

```bash
bash code/chapter04_attention/scripts/run_real_structure.sh
```

脚本配置 `code/chapter01_tensor_loader`，构建 `gguf_first_layer_inspect`，再运行 `real_tinyllama_first_layer_structure_test`。通过条件是首层九个权重的名称、形状、数据类型和 payload 都符合 Llama 布局；该测试不执行量化 Linear。随后进入 [`连接真实 TinyLlama 解码器`](./06_real_tinyllama_decoder.md) 理解 Model/Context，完整数值执行由 Chapter05 验收。
