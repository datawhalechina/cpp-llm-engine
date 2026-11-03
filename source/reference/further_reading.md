# Further Reading 进一步阅读

完成主线后，可以按系统实现、Serving、性能和模型交付四个方向继续阅读。

## 1. 加深主线

- 回看 `chapter02_tensor_ops`，重点关注 view 语义和存储共享
- 回看 `chapter03_matmul`，重点关注算子边界和后端拆分
- 回看 `chapter04_attention`，重点关注 decoder 结构
- 回看 `chapter06_generation`，重点关注 runtime state、KV Cache 和 sampling

## 2. 走向 vLLM-like serving

- 阅读 `reference/ecosystem/vllm.md`
- 对照 request scheduling、batching、streaming 和 cache management
- 想清楚哪些部分值得迁移到自己的 runtime

## 3. 继续看性能

- 阅读 `reference/ecosystem/tensorrt_llm.md`
- 回看 `chapter07_benchmark`，重点关注测量和比较纪律
- 优化前先分清正确性、性能和 profiling

## 4. 扩展模型或交付视角

- 阅读 `reference/ecosystem/llama_cpp.md`
- 阅读 `reference/ecosystem/ollama.md`
- 阅读 `reference/ecosystem/gemma.md`

## 5. 理解训练框架和推理框架的分工

- 阅读 `reference/framework_comparison.md`
- 先分清训练、推理和本教程各自负责哪一层
- 再回到主线章节看对象、算子和 runtime 的拆分

## 6. 补背景

- `guides/cmake_guide.md`
- `guides/setup_cuda.md`
- `guides/gguf_parser.md`
- `reference/faq.md`
