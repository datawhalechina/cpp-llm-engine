# chapter06_generation 代码导航

本章代码包含 Request、TokenHistory、Sampler、模型执行器契约、Runtime 和单请求验证。

## 学习入口

- [Chapter06 新导读](../chapter06_generation/next/intro.md)
- [Prefill 与 Decode](../chapter06_generation/next/02_prefill_decode.md)
- [单请求循环](../chapter06_generation/next/03_single_request_loop.md)
- [词元历史与模型 K/V](../chapter06_generation/next/06_single_request_kv_semantics.md)

## 阅读顺序

1. 先看 `request.cpp` 和 `runtime.cpp`。
2. 再看 `model_executor.h`、`sampler.cpp` 和 `token_history.cpp`。
3. 最后看 Generation 测试、真实模型脚本和回调示例。

## 运行时骨架

| 文件 | 作用 |
| --- | --- |
| <a href="https://github.com/datawhalechina/cpp-llm-engine/blob/main/code/chapter06_generation/include/chapter06_generation/request.h"><code>request.h</code></a> | 请求状态接口 |
| <a href="https://github.com/datawhalechina/cpp-llm-engine/blob/main/code/chapter06_generation/src/request.cpp"><code>request.cpp</code></a> | 请求生命周期 |
| <a href="https://github.com/datawhalechina/cpp-llm-engine/blob/main/code/chapter06_generation/include/chapter06_generation/runtime.h"><code>runtime.h</code></a> | Generation Core 接口 |
| <a href="https://github.com/datawhalechina/cpp-llm-engine/blob/main/code/chapter06_generation/src/runtime.cpp"><code>runtime.cpp</code></a> | Prefill、Decode、采样和停止编排 |
| <a href="https://github.com/datawhalechina/cpp-llm-engine/blob/main/code/chapter06_generation/include/chapter06_generation/token_history.h"><code>token_history.h</code></a> | 请求可观察的词元历史，不是模型 K/V |
| <a href="https://github.com/datawhalechina/cpp-llm-engine/blob/main/code/chapter06_generation/include/chapter06_generation/model_executor.h"><code>model_executor.h</code></a> | Prefill、Decode 和模型状态契约 |
| <a href="https://github.com/datawhalechina/cpp-llm-engine/blob/main/code/chapter06_generation/src/sampler.cpp"><code>sampler.cpp</code></a> | 采样实现 |
| <a href="https://github.com/datawhalechina/cpp-llm-engine/blob/main/code/chapter06_generation/src/transformer.cpp"><code>transformer.cpp</code></a> | 确定性教学执行器 |

## 验证与脚本

| 文件 | 作用 |
| --- | --- |
| <a href="https://github.com/datawhalechina/cpp-llm-engine/blob/main/code/chapter06_generation/examples/main_generation.cpp"><code>main_generation.cpp</code></a> | 单请求生成示例 |
| <a href="https://github.com/datawhalechina/cpp-llm-engine/blob/main/code/chapter06_generation/examples/main_streaming.cpp"><code>main_streaming.cpp</code></a> | 单请求逐词元回调示例 |
| <a href="https://github.com/datawhalechina/cpp-llm-engine/blob/main/code/chapter06_generation/tests/request_contract_test.cpp"><code>request_contract_test.cpp</code></a> | Request 状态机契约 |
| <a href="https://github.com/datawhalechina/cpp-llm-engine/blob/main/code/chapter06_generation/tests/chapter06_generation_test.cpp"><code>chapter06_generation_test.cpp</code></a> | Generation Core 组件测试 |
| <a href="https://github.com/datawhalechina/cpp-llm-engine/blob/main/code/chapter06_generation/tests/token_history_test.cpp"><code>token_history_test.cpp</code></a> | TokenHistory 顺序和清空测试 |
| <a href="https://github.com/datawhalechina/cpp-llm-engine/blob/main/code/chapter06_generation/scripts/run_generation_core.sh"><code>run_generation_core.sh</code></a> | 默认 CPU 入口 |
| <a href="https://github.com/datawhalechina/cpp-llm-engine/blob/main/code/chapter06_generation/scripts/run_real_generation.sh"><code>run_real_generation.sh</code></a> | 真实 TinyLlama 单请求入口 |

## 相邻章节

- [Chapter07 K/V Cache 代码导航](./chapter07_kv_cache.md)
- [Chapter08 Scheduling 代码导航](./chapter08_scheduling.md)
