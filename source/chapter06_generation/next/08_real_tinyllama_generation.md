# 06.8 验证真实 TinyLlama 单请求生成

合成 logits 可以稳定验证状态和停止条件，但不能证明 Tokenizer、22 层 Decoder、量化权重和模型 K/V 已经连通。真实模型闭环保持同一个 `Request -> Runtime -> Sampler` 控制流，只把模型执行器替换为 TinyLlama Context。

## 文本怎样进入 Generation Core

真实入口由 `RealLlamaEngine` 组装，执行顺序如下：

| 顺序 | 对象 | 可观察结果 |
| --- | --- | --- |
| 1 | 真实 Tokenizer | prompt 文本变为 TinyLlama token IDs |
| 2 | `RealLlamaExecutorFactory` | 加载一次只读权重，为请求创建独立 Context |
| 3 | `Runtime::generate()` | 执行 Prefill、Greedy、Decode 和停止判断 |
| 4 | 真实 Tokenizer | 生成 token IDs 还原为文本 |

可执行入口位于 <a href="/_code/engine/examples/real_tinyllama_generation_smoke.cpp"><code>real_tinyllama_generation_smoke.cpp</code></a>。同一个 Engine 连续生成两次时会创建两份请求 Context，权重可以共享，位置和模型 K/V 不能共享。

## 固定模型与结果

教程使用 `TinyLlama-1.1B-Chat-v1.0` 的 Q4_K_M GGUF。固定 prompt 和 Greedy 后，当前 CPU reference 保存以下 golden：

| 条件 | 固定值 |
| --- | --- |
| prompt | `Hello TinyLlama` |
| prompt token 数 | 6 |
| 最大输出数 | 4 |
| 生成 token IDs | `29892 306 29915 29885` |
| 解码文本 | `, I'm` |
| 请求隔离 | 连续两次结果一致 |

Golden 用于发现 Loader、Tokenizer、量化 Linear、Attention 或 Runtime 的行为变化，不用于评价回答质量。模型版本、Tokenizer 或数值 backend 改变后，应先定位差异来源，再决定是否更新结果。

## 运行真实模型验收

确认模型位于 `llama.cpp/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf`，然后执行：

```bash
cmake -S code/engine -B /tmp/cpp-llm-real-chapter06 \
  -DCHAPTER01_ENABLE_LLAMA_CPP=ON \
  -DLLAMA_CPP_ROOT="$PWD/llama.cpp" \
  -DCHAPTER05_SOURCE_DIR="$PWD/code/chapter05_quantization" \
  -DCPP_LLM_ENGINE_REAL_MODEL="$PWD/llama.cpp/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf" \
  -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/cpp-llm-real-chapter06 \
  --target real_tinyllama_generation_smoke -j2
ctest --test-dir /tmp/cpp-llm-real-chapter06 \
  -R '^real_tinyllama_generation_test$' --output-on-failure
```

通过输出必须同时包含 `golden_tokens_match: true` 和 `context_isolation_match: true`。前者验证单请求数值链，后者验证共享权重时两次请求没有复用上一次的模型状态。
