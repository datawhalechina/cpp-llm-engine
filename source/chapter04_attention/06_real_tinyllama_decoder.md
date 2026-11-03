# 04.6 运行真实 TinyLlama 解码器

Chapter01 已把 TinyLlama GGUF 转换为模型配置、张量映射和 Llama 权重布局。本节消费这些对象，执行 22 个解码器块、最终 RMSNorm 和 LM Head，并把候选分数（logits）与 llama.cpp 基线比较。

## 从权重布局到多层解码器

TinyLlama 的 `block_count=22` 对应 `blk.0` 到 `blk.21`。每个解码器块包含注意力和 FFN 的九个权重张量；`token_embd.weight`、`output_norm.weight` 和 `output.weight` 位于解码器块堆栈之外。

![TinyLlama 22 个解码器块的层级结构](../assets/diagrams/chapter01-tinyllama-decoder-hierarchy.svg)

真实执行代码独立放在 <a href="../_code/model/src/real_llama_cpu_decoder.cpp"><code>code/model/real_llama_cpu_decoder.cpp</code></a>，避免 Chapter01 模型加载器和 Chapter04 教学算子互相承担对方职责。`RealLlamaCpuModel` 持有共享只读权重，`RealLlamaCpuContext` 保存单个请求的位置、分层 K/V 和注意力统计。

| 对象 | 保存内容 | 生命周期 |
| --- | --- | --- |
| `RealLlamaCpuModel` | 词元嵌入、22 层权重、最终归一化、LM Head | 模型加载到服务卸载 |
| `RealLlamaCpuContext` | 位置、每层 K/V、注意力统计 | 单个请求或序列 |
| Chapter06 `Request` | 输入文本、生成词元、阶段和停止原因 | 准入到终态清理 |

同一个模型可以创建多个执行上下文。销毁请求只释放所属上下文，不卸载共享权重，也不影响其他请求的 K/V。

## 数值对照与请求隔离

完整连通性检查使用同一组 6 个词元分别运行自研 CPU 解码器和 llama.cpp。llama.cpp 参考路径关闭 Flash Attention，并使用 F32 K/V Cache，使双方的注意力精度设置更接近。

| 指标 | 当前结果 | 验收门槛 |
| --- | --- | --- |
| 余弦相似度 | `0.99952` | 不低于 `0.999` |
| 平均绝对误差 | `0.117456` | 不高于 `0.15` |
| RMSE | `0.148398` | 不高于 `0.20` |
| Top-1 | 词元 `29892`，双方一致 | 候选一致 |
| Top-5 重合数 | `5/5` | 至少 `4/5` |
| 4 词元贪心序列 | `29892 306 29915 29885` | 双方序列一致 |

22 层会累积不同矩阵乘加顺序造成的浮点误差，因此不能只用单个最大绝对误差判断完整模型。整体方向、平均误差、候选集合和连续贪心结果共同组成数值证据。

## 验证与衔接

默认模型路径为 `llama.cpp/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf`。已准备 GGUF 和 llama.cpp 源码后运行：

```bash
bash scripts/run_real_decoder_learning.sh
```

脚本先验证 Chapter04 教学解码器和 Chapter05 量化后端，再运行 22 层候选分数对照，最后构建引擎的真实 TinyLlama 生成连通性检查。对照程序检查余弦相似度、平均绝对误差、RMSE 和 Top-5；CTest 再确认固定贪心词元、两个执行上下文的输出一致、请求完成以及缓存状态互不污染。

这些结果向 [`Chapter06`](../chapter06_generation/intro.md) 提供可执行的解码器上下文和候选分数；生成运行时只推进请求，不重新实现模型数学。
