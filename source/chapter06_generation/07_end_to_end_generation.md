# 06.7 验证端到端生成链路

组件测试分别验证 Request、Executor、Sampler 和 Scheduler，端到端测试则固定 prompt、模型、采样策略和 token budget，确认它们能组成可重复生成链路。这里验证的是执行闭环，不评价文本质量或服务吞吐。

## 生成链路的测试契约

最小测试从确定性 Greedy 开始。输入经过 prefill 得到首个 logits，随后重复 sample、append 和 decode，直到 EOS、token budget、context limit 或执行错误进入终态。

| 检查项 | 固定条件 | 通过标准 |
| --- | --- | --- |
| 输入 | prompt token 固定 | prefill 只执行一次 |
| 采样 | Greedy、相同 logits | token 序列可重复 |
| 状态推进 | max tokens、EOS 配置固定 | finish reason 与输出长度一致 |
| 请求隔离 | 两个独立 Executor/Context | token history 与分层 K/V 不串扰 |
| 失败路径 | prefill/decode 返回错误或空 logits | Request 进入 `Failed`，不生成占位 token |

仓库使用轻量命名检查和 CTest，不依赖 GoogleTest。Request 状态转换位于 <a href="../_code/chapter06_generation/tests/request_contract_test.cpp"><code>request_contract_test.cpp</code></a>，完整生成链路位于 <a href="../_code/chapter06_generation/tests/chapter06_generation_test.cpp"><code>chapter06_generation_test.cpp</code></a>，多请求事务由 <a href="../_code/chapter06_generation/tests/execution_transaction_test.cpp"><code>execution_transaction_test.cpp</code></a> 验证。

## 真实 TinyLlama 生成证据

教学 smoke 稳定后，`RealTokenizerBackend` 负责文本与 token 转换，`RealLlamaExecutorFactory` 共享 22 层模型权重，并为每个请求创建独立 `RealLlamaCpuContext`。`Runtime::generate()` 的 Request、Sampler 和停止规则保持不变，只替换 logits 来源。

从仓库根目录执行：

```bash
cmake -S code/engine -B /tmp/cpp-llm-engine-real-generation \
  -DCHAPTER01_ENABLE_LLAMA_CPP=ON \
  -DLLAMA_CPP_ROOT="$PWD/llama.cpp" \
  -DCHAPTER05_SOURCE_DIR="$PWD/code/chapter05_quantization" \
  -DCPP_LLM_ENGINE_REAL_MODEL="$PWD/llama.cpp/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf" \
  -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/cpp-llm-engine-real-generation \
  --target real_tinyllama_generation_smoke -j2
/tmp/cpp-llm-engine-real-generation/real_tinyllama_generation_smoke \
  llama.cpp/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf \
  4 "Hello TinyLlama"
ctest --test-dir /tmp/cpp-llm-engine-real-generation \
  -R real_tinyllama_generation_test --output-on-failure
```

| 检查项 | 固定结果 |
| --- | --- |
| prompt | `Hello TinyLlama`，6 个 token |
| 生成 token | `29892 306 29915 29885` |
| detokenize 文本 | `, I'm` |
| golden CTest | `golden_tokens_match: true` |
| 连续请求隔离 | `context_isolation_match: true` |
| 停止条件 | `max_new_tokens=4`，或 EOS 提前结束 |

模型文件、tokenizer 或数值 backend 改变后，golden 只能在确认变化原因后更新，不能用新输出直接覆盖失败证据。

## 验证与衔接

```bash
bash code/chapter06_generation/scripts/run.sh
```

失败时按最早可观察边界定位：无首个 logits 先查 prefill；输出长度异常查停止条件；单请求稳定但双请求异常查 Context 隔离；token 与预期不同则依次比较 logits、Sampler 和量化 backend。测试通过后阅读 [`06.8 Runtime 框架对照`](./08_vllm_comparison.md)，再进入 Request 生命周期与 Continuous Batching。
