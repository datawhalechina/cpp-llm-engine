# 06.2 组装多层 Transformer

多层 Decoder 共享 hidden state 的输入输出契约，但每层持有独立参数。Chapter06 需要验证 runtime 能循环调用多层模型；真实 TinyLlama 的 22 层参数和数学计算由 Chapter01 的 CPU model/context 实现。

源码阅读先看 <a href="../_code/chapter06_generation/src/transformer.cpp"><code>transformer.cpp</code></a> 的层循环，再看 <a href="../_code/chapter06_generation/include/chapter06_generation/model_executor.h"><code>model_executor.h</code></a> 如何把模型计算交给 Runtime。

## 多层结构与两种执行模型

标准 Transformer 的每层都有独立的 norm、Attention 和 FFN 权重，但输入输出 hidden shape 保持一致。第 0 层输出直接成为第 1 层输入，依次通过全部层；残差连接位于每个 DecoderLayer 内部。

```text
embedding
  -> layer 0: RMSNorm -> Attention -> residual -> RMSNorm -> FFN -> residual
  -> layer 1: RMSNorm -> Attention -> residual -> RMSNorm -> FFN -> residual
  -> ...
  -> final norm -> LM Head -> logits
```

这条结构解释了两项工程约束：每层权重不能共享，临时 hidden buffer 可以在层间复用。TinyLlama 的 22 层都是同一种结构，不是 22 种不同算子。

Chapter06 smoke `Transformer` 创建 N 个只保存 `layer_index` 的 `DecoderLayer`。每层给向量增加一个确定性 bias，用于证明循环顺序和输出推进；它没有 QKV、RMSNorm、FFN 或模型权重，不能作为模型数值实现。

真实路径的 `RealLlamaCpuModel` 共享 GGUF 权重，`RealLlamaCpuContext` 保存请求独占的 position 和 22 层 K/V。Engine adapter 把该 Context 暴露为 `RequestModelExecutor`，Chapter06 `Runtime` 只调用 `prefill_chunk()` 和 `decode()`。

| 层次 | Chapter06 smoke model | TinyLlama CPU model |
| --- | --- | --- |
| 层对象 | `DecoderLayer(layer_index)` | 每层九个 GGUF 权重 |
| 层计算 | 对每个值增加确定性 bias | RMSNorm、GQA Attention、SwiGLU FFN、残差 |
| 状态 | Request token history | position 与 22 层 K/V |
| 目的 | 验证 runtime 编排 | 验证真实模型生成 |

## Runtime 接口

模型实现通过窄接口进入 generation runtime。支持 chunked prefill 的 Executor 可以分段消费 prompt，只有 final chunk 返回 logits；decode 每次消费上一个 token，并推进同一个请求 Context。

```cpp
class RequestModelExecutor {
public:
    virtual bool prefill_chunk(/* chunk and boundary flags */) = 0;
    virtual bool decode(TokenId token, Logits* logits, std::string* error) = 0;
};
```

上面的代码省略了完整参数，接口定义以 `model_executor.h` 为准。关键设计是 Runtime 编排生命周期，Executor 负责模型计算；替换模型实现不改变 Request、Sampler 和 Scheduler。

## 验证与衔接

smoke 测试检查两层对象确实被循环调用，并验证 prefill、decode、EOS 和 token budget。真实模型测试检查固定 TinyLlama GGUF 的 22 层输出、golden token、分层 K/V 增长和请求 Context 隔离。

| 检查项 | 对应证据 |
| --- | --- |
| N 层循环和组件 ready | `chapter06_generation_test.cpp` |
| Runtime 与模型实现解耦 | `model_executor.h`、`runtime.cpp` |
| 22 层真实 Decoder | `real_llama_cpu_decoder.cpp` |
| 固定 prompt 的真实生成 | `real_tinyllama_generation_smoke` |

验证模型执行与 Runtime 接口解耦后，进入 [`06.3 朴素生成`](./03_naive_generation.md)，把一次 logits 计算接入逐 token 循环。
