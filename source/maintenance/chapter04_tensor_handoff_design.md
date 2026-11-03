# Chapter02 到 Chapter04 的 Attention Tensor 交接规格

Chapter02 已能承载 1–4 维连续与非连续 View。Chapter04 的教学 Decoder 仍使用独立二维 Tensor，CPU Attention backend 则通过 `AttentionRequest` 读取连续指针。第一轮接入增加适配器，不同时重写 Linear、Norm、FFN 和 DecoderLayer。

## 输入布局与执行布局

投影输出先使用三维 token-major 布局；增加 batch 后在最前面添加一维。最后两个轴保持 `[heads, head_dim]`，因此每个 token 的全部 head 数据在内存中连续。

| 对象 | rank 3 | rank 4 |
| --- | --- | --- |
| Query | `[query_tokens, heads, head_dim]` | `[batch, query_tokens, heads, head_dim]` |
| Key/Value | `[key_tokens, heads, head_dim]` | `[batch, key_tokens, heads, head_dim]` |
| Output | 与 Query 相同 | 与 Query 相同 |

Attention score 的逻辑 shape 仍是 `[batch, heads, query_tokens, key_tokens]`，但 reference kernel 不需要长期保存完整 score Tensor。适配器只把连续 Q/K/V View 转为当前 backend 使用的指针和维度。

## 适配器契约

适配器接收三个只读 `TensorViewRef`、一个可写 `MutableTensorViewRef` 和 `AttentionBackend`。返回 Chapter02 Tensor 的兼容入口在内部创建输出 View。两条入口按以下顺序验证：

1. Storage 指针和 View 范围有效；
2. Q/K/V rank 全部为 3 或全部为 4；
3. dtype 为 Float32、device 为 CPU、布局连续；
4. batch、heads 和 head_dim 一致；
5. Key 与 Value 的 token 数一致；
6. causal offset 不溢出可见位置。

rank 3 只调用一次 backend。rank 4 按 batch 计算每个样本的起始指针，逐个调用现有 backend；该循环先固定 batch 语义，不承诺并行执行。

```text
TensorViewRef Q/K/V
        ↓ rank、shape、layout 检查
每个 batch 生成 AttentionRequest
        ↓
AttentionBackend::compute
        ↓
调用方提供的连续可写 View
```

Chapter02 Tensor 增加 `data()`：只有 ready 且连续时返回当前 View 首元素指针，非连续 View 返回空指针。调用者必须显式执行 `contiguous()`，适配器不隐藏复制。

## 与现有 Chapter04 对象的关系

第一轮保留以下两条入口：

| 入口 | 用途 | 当前消费者 |
| --- | --- | --- |
| Chapter04 二维 `Tensor` | 展示 Linear、Norm、Residual、FFN 和完整 DecoderLayer | `MultiHeadAttention::forward()`、`DecoderLayer` |
| 公共 N 维 View 适配器 | 固定 Q/K/V、输出和 batch 的真实布局契约 | 跨章 Attention 测试，后续 Decoder 重构 |

公共 View 已覆盖 Linear、LayerNorm、ReLU、Residual Add、FFN、二维 MultiHeadAttention 和完整 DecoderLayer。Chapter04 二维 Tensor 继续承担教学 Storage 和兼容入口，不再作为跨章接口。KV Cache 尚未接入教学 DecoderLayer，后续缓存迁移必须与当前无状态 forward 分开验证。

## 验证矩阵

| 场景 | 输入 | 预期证据 |
| --- | --- | --- |
| 3D causal prefill | Q/K/V `[2,1,1]` | 第一个 query 只读取第一个 value |
| 3D decode | Q `[1,H,Dh]`、K/V `[S,H,Dh]` | position offset 可以读取历史位置 |
| 4D batch | `[2,T,H,Dh]` | 两个 batch 输出互不串数据 |
| 非连续 View | transpose 后直接提交 | 明确拒绝；连续化后成功 |
| shape 错误 | batch/head/head_dim/token 不一致 | 在 backend 前返回 `Status` |
| 设备和类型错误 | CUDA 或非 Float32 contract | 在 backend 前返回 `Status` |
| backend 失败 | 注入返回错误的 CPU backend | 已有输出保持不变 |

验证顺序为适配器单测、Chapter04 章节脚本、Chapter00–09 全局回归。学习者正文只在测试通过后增加 3D/4D 执行布局和跨章入口。
