# Chapter 06：单请求 Generation Core

本目录把模型执行器连接成 `prefill -> decode -> sampling -> stop` 的单请求生成循环。Chapter07 管理模型 K/V 的物理存储，Chapter08 选择多请求批次，Chapter10 管理服务生命周期。学习说明见 [`Chapter06 导读`](../../source/chapter06_generation/next/intro.md)。

## 代码架构

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

`TokenHistory` 只记录可观察的 token id，不保存 Attention K/V。模型执行器持有单请求模型状态；Chapter07 管理分页 KV 资源；Chapter08 决定哪些请求组成下一步 batch。Chapter06 只定义一次请求怎样前进，不决定跨请求顺序。

## 阅读顺序与资产

| 资产 | 作用 |
| --- | --- |
| `src/request.cpp` | Request 状态、取消、超时和终态 |
| `src/token_history.cpp` | Request 可观察的词元历史；不保存 Attention K/V |
| `model_executor.h` | 真实模型 Prefill、Decode、状态快照和批量执行契约 |
| `src/sampler.cpp` | 从 logits 选择下一词元 |
| `src/runtime.cpp` | 模型调用、状态、采样、停止和回调编排 |
| `embedding.cpp`、`position_encoding.cpp`、`decoder_layer.cpp`、`transformer.cpp` | 不依赖模型文件的确定性教学执行器 |
| `tests/` | Request、TokenHistory、Sampler 和 Generation Core 测试 |
| `examples/main_generation.cpp` | 普通生成示例 |
| `examples/main_streaming.cpp` | 单请求逐词元回调示例 |
| `scripts/run_generation_core.sh` | 默认 CPU 构建和验证入口 |

先读 `request.h` 的状态，再读 `model_executor.h` 的模型调用接口；随后沿 `Runtime::prefill_chunk()` 和 `Runtime::decode_one()` 看一次状态提交，最后阅读批量入口。`commit_batch_logits()` 会先校验全部请求和 logits 行，再统一提交，避免半个 batch 已写入、半个 batch 失败；Scheduler 只负责选择哪些请求进入该 batch。

## 运行与验证

默认 CPU 入口：

```bash
bash code/chapter06_generation/scripts/run_generation_core.sh
```

验证重点是请求状态、Prefill/Decode、采样与停止、词元历史、回调时序和模型错误传播。真实 TinyLlama 单请求入口：

```bash
bash code/chapter06_generation/scripts/run_real_generation.sh
```

## 扩展入口

新模型通过 `RequestModelExecutor` 接入，并在返回成功前保证 logits 非空。新的 KV 存储策略进入 Chapter07，新的优先级或 batching 策略进入 Chapter08，HTTP/SSE 与取消来源进入 Chapter10；这些能力不应加入 `Runtime` 的单请求状态机。
