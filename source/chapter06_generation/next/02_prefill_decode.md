# 06.2 区分预填充与逐词元解码

同一个请求会以两种形状进入模型。预填充（Prefill）一次接收多个提示词元并建立上下文；逐词元解码（Decode）每次接收一个已生成词元，在已有上下文后追加计算。二者都产生候选分数（logits），但输入规模、模型状态和调用次数不同。

## 什么是生成运行时

“运行时”描述程序运行期间组织动态工作的组件。模型对象保存配置和只读权重，模型执行器完成一次预填充或解码前向；生成运行时（`Runtime`）把这些单次调用连接成一个请求的完整生成过程：检查请求状态，调用执行器，调用采样器，提交新词元，并判断停止或失败。

限定词决定这个词的具体含义：C++ 运行时提供语言执行所需的底层支持，CUDA 运行时提供设备 API，而本章的生成运行时是教程定义的 C++ 组件。看到源码中的 `Runtime` 时应按本章职责理解，而不是把所有“程序运行期间的代码”都归入这个类。

在本教程中，生成运行时聚焦单请求控制流。它通过执行器使用 K/V 状态，但不决定缓存的连续或分页布局；它推进一个请求，但不选择多个请求的执行顺序；它产生词元和回调，但不处理 HTTP/SSE 协议。这三项职责分别由 Chapter07 缓存、Chapter08 调度器和 Chapter10 服务层承担。

```text
请求状态 -> 生成运行时 -> 模型执行器 -> 模型 + 执行上下文
               |              |
               v              v
             采样器         候选分数
               |
               v
          新词元 / 请求终态
```

## 两种模型调用

下面用 prompt `[1, 2, 3]` 和两个生成词元说明调用顺序。首个输出词元直接从 Prefill 返回的 logits 采样，因此第一次 Decode 消费的是这个刚生成的词元。

| 顺序 | 模型调用 | 本次输入 | 调用后的已提交模型位置 | 产生的结果 |
| --- | --- | --- | --- | --- |
| 1 | 预填充 | `[1, 2, 3]` | 3 | 第一组待采样候选分数 |
| 2 | 采样 | 预填充候选分数 | 3 | 第一个生成词元 `g0` |
| 3 | 逐词元解码 | `g0` | 4 | 第二组待采样候选分数 |
| 4 | 采样 | 解码候选分数 | 4 | 第二个生成词元 `g1` |

`Request::generated()` 记录已经交给调用者的输出，模型执行器的 `committed_tokens` 记录已经进入模型执行上下文的输入位置。采样出 `g1` 后，如果请求立即结束，`g1` 不需要再次进入模型；因此两个计数可以不同。

## 生成运行时怎样提交结果

<a href="https://github.com/datawhalechina/cpp-llm-engine/blob/main/code/chapter06_generation/src/runtime.cpp"><code>runtime.cpp</code></a> 中的 `prefill_chunk()` 依次完成以下动作：

1. 检查上下文预算、终态和本次 chunk 大小。
2. 调用执行器，把成功消费的 prompt 数写入 `prefill_offset`。
3. 最后一段 Prefill 返回 logits 后，采样并提交首个生成词元。

后续 `decode_one()` 读取最后一个生成词元，调用执行器并提交下一次采样结果。只有模型调用成功且 logits 非空时，请求才增加输出；执行器失败会让请求进入 `Failed`，不会伪造占位词元。

| 可观察对象 | Prefill 后 | 一次 Decode 后 |
| --- | --- | --- |
| `prefill_offset` | 等于 prompt 长度 | 不再变化 |
| `generated_size` | 增加 1 | 再增加 1 |
| `phase` | `Decoding` 或 `Finished` | `Decoding` 或 `Finished` |
| 模型 Context | 保存 prompt 的分层 K/V | 追加上一个生成词元的 K/V |
| 教学词元历史 | prompt 加首个输出 | 再追加一个输出 |

Chunked Prefill 只是把第一行拆成多次模型调用；如何决定每次允许消费多少 prompt 词元属于 Chapter08 Scheduling，模型 K/V 的分页存储属于 Chapter07。

## 验证阶段变化

执行单请求生成核心测试：

```bash
cmake -S code/chapter06_generation -B /tmp/cpp-llm-chapter06-core \
  -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/cpp-llm-chapter06-core \
  --target chapter06_generation_test -j2
ctest --test-dir /tmp/cpp-llm-chapter06-core \
  -R '^chapter06_generation_test$' --output-on-failure
```

测试固定三个 prompt 词元和三个输出词元，检查 Prefill 只提交一次、请求从 `Prefilled` 进入 `Finished`，教学词元历史最终包含六个词元；超出上下文预算的请求必须在模型执行前进入 `Rejected`。下一篇把 Prefill、采样和 Decode 连接成单请求循环。
