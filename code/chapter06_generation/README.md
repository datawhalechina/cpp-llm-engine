# Chapter 06：单请求 Generation Core

本目录把模型执行器连接成 `prefill -> decode -> sampling -> stop` 的单请求生成循环。Chapter07 管理模型 K/V 的物理存储，Chapter08 选择多请求批次，Chapter10 管理服务生命周期。学习说明见 [`Chapter06 导读`](../../source/chapter06_generation/next/intro.md)。

## 代码架构

调用方拥有 `Request`、`Sampler` 和请求专属 `RequestModelExecutor`，这些对象必须覆盖一次生成调用。`Runtime` 只保存配置和可选教学 `Transformer` 的非拥有指针；模型执行器持有位置与模型 K/V，`TokenHistory` 始终记录已经提交给请求的 token ID。模型失败先进入 Request 终态，batch logits 则先校验所有行再统一提交。

`ExecutorBatchRunner` 还提供请求绑定、缓存前缀采用、缓存前缀发布和批量 Decode 交接点。Chapter06 只定义这些生命周期契约；Chapter07 保存共享 K/V，Engine Factory 同步真实模型 Context，Chapter10 决定准入与释放时机。

```text
Request(prompt + phase + generated + token history)
   │
   ├─ prefill ──> RequestModelExecutor / 教学 Transformer ──> logits
   │                                                        │
   └─ decode  <──────────────────── 上一个生成 token        ▼
                                                        Sampler
                                                           │
                                                           ▼
                                                token + stop / failure
```

`TokenHistory` 只记录可观察的 token ID，不保存 Attention K/V。Chapter07 管理分页 KV 资源；Chapter08 决定哪些请求组成下一步 batch。

真实模型适配器位于 `code/engine`，它们复用同一组 Chapter06 接口，但各自持有不同的模型状态：

| Factory | 模型 Context | Chapter06 观察到的公共结果 |
| --- | --- | --- |
| `RealLlamaExecutorFactory` | TinyLlama/Qwen3 的标准分层 K/V | Prefill/Decode logits、已提交位置和失败状态 |
| `RealQwen2MoeExecutorFactory` | Qwen1.5-MoE 的 24 层 K/V 与专家执行状态 | 相同 Runtime 状态和批量 Decode 结果 |
| `RealDeepSeek2ExecutorFactory` | DeepSeek-V2-Lite 的 27 层低秩 MLA 历史 | 相同 Runtime 状态和批量 Decode 结果 |

## 阅读顺序与资产

| 顺序 | 文件 | 阅读重点 | 对应验证 |
| --- | --- | --- | --- |
| 1 | `request.h`、`src/request.cpp` | Prefill 游标、缓存前缀采用、生成结果和五种吸收终态 | `chapter06_request_contract_test` |
| 2 | `model_executor.h` | 请求专属 Context、Prefill/Decode、缓存生命周期钩子和失败返回 | Generation 与 ServiceCore 契约测试 |
| 3 | `src/sampler.cpp`、`src/token_history.cpp` | Greedy 选择与可观察 token 历史 | sampler、token history unit 测试 |
| 4 | `src/runtime.cpp` | 模型调用、提交、停止与回调顺序 | `chapter06_generation_test` |
| 5 | `embedding.cpp`、`position_encoding.cpp`、`decoder_layer.cpp`、`transformer.cpp` | 无模型文件的确定性教学执行器 | `chapter06_generation_demo` |
| 6 | `code/engine/examples/real_tinyllama_generation_smoke.cpp` | TinyLlama/Qwen3 Tokenizer、完整 Decoder 和 Context 隔离 | 两种稠密架构的 generation smoke |
| 7 | `code/engine/src/real_qwen2_moe_executor_factory.cpp` | 24 层 MoE Context、分层 K/V 与 Runtime 适配 | Qwen1.5-MoE Context、Runtime 和文本 smoke |
| 8 | `code/engine/src/real_deepseek2_executor_factory.cpp` | 27 层 MLA Context、混合稠密/MoE 层与 Runtime 适配 | DeepSeek Runtime、文本与 batch smoke |

先读 `request.h` 的状态，再读 `model_executor.h` 的模型调用接口；随后沿 `Runtime::prefill_chunk()` 和 `Runtime::decode_one()` 看一次状态提交，最后阅读批量入口。`commit_batch_logits()` 会先校验全部请求和 logits 行，再统一提交，避免半个 batch 已写入、半个 batch 失败；Scheduler 只负责选择哪些请求进入该 batch。

## 运行与验证

默认 CPU 入口：

```bash
bash code/chapter06_generation/scripts/run_generation_core.sh
```

示例应输出 Prefill 数量、生成步数、词元历史长度和 `request phase: finished`。CTest 按以下层级提供证据：

| 层级 | 测试 | 证明内容 |
| --- | --- | --- |
| unit | sampler、token history | 最大 logits 选择、空输入失败和历史顺序 |
| contract | request contract | Prefill 进度和五种吸收终态 |
| component | generation test | Prefill/Decode、停止、回调、batch 提交和模型错误 |
| real_model | TinyLlama、Qwen3、Qwen1.5-MoE、DeepSeek generation | 固定 token、文本结果和连续请求 Context 隔离 |

真实 TinyLlama 单请求入口：

```bash
bash code/chapter06_generation/scripts/run_real_generation.sh
```

Qwen3 使用同一稠密模型入口，通过 GGUF `general.architecture` 自动选择结构：

```bash
bash code/chapter06_generation/scripts/run_qwen3_generation.sh
```

Qwen1.5-MoE 按三个层级验证请求状态、Runtime 和文本输入：

```bash
bash code/chapter06_generation/scripts/run_qwen15_moe_context.sh
bash code/engine/scripts/run_qwen15_moe_runtime.sh
bash code/engine/scripts/run_qwen15_moe_generation.sh
```

第一条直接检查 24 层 Context 的位置和 K/V；第二条用 token ID 检查 Chapter06 状态机；第三条增加真实 Tokenizer 与文本解码。模型权重按层加载，命令会读取约 7.4 GB GGUF，运行时间高于默认 TinyLlama 路线。

DeepSeek-V2-Lite 使用相同 Runtime 验证 27 层压缩 MLA 历史、固定词元和文本解码：

```bash
bash code/engine/scripts/run_deepseek_v2_lite_generation.sh
```

## 扩展入口

新模型通过 `RequestModelExecutor` 接入，并在返回成功前保证 logits 非空。新的 KV 存储策略进入 Chapter07，新的优先级或 batching 策略进入 Chapter08，HTTP/SSE 与取消来源进入 Chapter10；这些能力不应加入 `Runtime` 的单请求状态机。
