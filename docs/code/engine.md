# engine First Success 代码导航

Engine 将 prompt、token tensor、Request、真实 Model/Context、Chapter 06 Runtime 和输出连接成 CPU 纵向执行路径。

## 阅读顺序

1. 先看 `engine.h`，理解 `TokenTensor`、`GenerationResult` 和 `Engine` 边界
2. 再看 `tokenizer_backend.h`，理解 toy/real tokenizer 的替换边界
3. 再看 `code/model` 的 `real_llama_cpu_decoder.h`，理解只读 Model 与请求 Context 的所有权
4. 阅读 `real_llama_executor_factory.h`，确认一个 Factory 如何为多个请求创建独立 Executor/Context
5. 对照 `engine.cpp` 和 `real_llama_engine.cpp`，跟踪 toy/real 两条生成路径
6. 最后运行 demo、真实生成/服务/HTTP smoke 和 CTest

## 核心代码

| 文件 | 作用 |
| --- | --- |
| <a href="/_code/engine/include/cpp_llm_engine/engine.h"><code>engine.h</code></a> | 仓库级 Engine 接口 |
| <a href="/_code/engine/include/cpp_llm_engine/tokenizer_backend.h"><code>tokenizer_backend.h</code></a> | tokenizer backend contract |
| <a href="/_code/engine/include/cpp_llm_engine/real_tokenizer_backend.h"><code>real_tokenizer_backend.h</code></a> | llama.cpp tokenizer 适配接口 |
| <a href="/_code/engine/include/cpp_llm_engine/real_llama_engine.h"><code>real_llama_engine.h</code></a> | 真实 TinyLlama 单请求生成接口 |
| <a href="/_code/engine/include/cpp_llm_engine/real_llama_executor_factory.h"><code>real_llama_executor_factory.h</code></a> | 共享 Model、创建每请求 Executor/Context |
| <a href="/_code/engine/include/cpp_llm_engine/real_llama_cuda_executor_factory.h"><code>real_llama_cuda_executor_factory.h</code></a> | CUDA Model 到 Chapter06 Runtime 的每请求适配 |
| <a href="/_code/model/include/cpp_llm_engine/model/real_llama_cpu_decoder.h"><code>real_llama_cpu_decoder.h</code></a> | 共享 22 层权重的 Model 与独占分层 K/V 的请求 Context |
| <a href="/_code/chapter01_tensor_loader/src/model_config.h"><code>model_config.h</code></a> | Chapter 01 提供的模型配置边界 |
| <a href="/_code/engine/src/engine.cpp"><code>engine.cpp</code></a> | token tensor、tokenize 和 generation bridge |
| <a href="/_code/engine/src/real_llama_engine.cpp"><code>real_llama_engine.cpp</code></a> | 真实 tokenizer、Decoder 与 Chapter06 Runtime bridge |
| <a href="/_code/engine/examples/real_tinyllama_serving_smoke.cpp"><code>real_tinyllama_serving_smoke.cpp</code></a> | Chapter10 真实双请求、K/V 资源释放验证 |
| <a href="/_code/engine/examples/real_tinyllama_http_adapter_smoke.cpp"><code>real_tinyllama_http_adapter_smoke.cpp</code></a> | 真实同步/SSE completion 与 metrics 验证 |
| <a href="/_code/engine/examples/real_tinyllama_serving_benchmark.cpp"><code>real_tinyllama_serving_benchmark.cpp</code></a> | Chapter10 请求级 Worker 的顺序/并行计时、吞吐和 JSON 输出 |
| <a href="/_code/engine/scripts/run_real_cuda_runtime.sh"><code>run_real_cuda_runtime.sh</code></a> | 固定 token prompt 的真实 CUDA Generation Runtime 验收 |
| <a href="/_code/engine/scripts/run_real_cuda_generation.sh"><code>run_real_cuda_generation.sh</code></a> | 真实 tokenizer 到 CUDA 生成和 detokenizer 的文本闭环 |
| <a href="/_code/engine/scripts/run_real_cuda_serving.sh"><code>run_real_cuda_serving.sh</code></a> | CUDA Executor 到 Chapter10 Completion/SSE 和终态释放闭环 |
| <a href="/_code/engine/scripts/run_real_serving_benchmark.sh"><code>run_real_serving_benchmark.sh</code></a> | 固定 Release 构建与结果归档入口 |
| <a href="/_code/engine/examples/main.cpp"><code>examples/main.cpp</code></a> | 命令行 First Success |
| <a href="/_code/engine/tests/engine_smoke_test.cpp"><code>tests/engine_smoke_test.cpp</code></a> | 纵向切片 smoke test |
| <a href="/_code/engine/CMakeLists.txt"><code>CMakeLists.txt</code></a> | 独立构建入口 |
| <a href="/_code/engine/README.md"><code>README.md</code></a> | 运行说明和边界声明 |

## 运行

```bash
cmake -S code/engine -B code/engine/build
cmake --build code/engine/build
code/engine/build/cpp_llm_engine_demo --prompt "hello" --max-new-tokens 4
ctest --test-dir code/engine/build --output-on-failure
```

默认命令验证 toy Engine。真实 TinyLlama 路径的构建命令和当前 greedy 输出见 <a href="/_code/engine/README.md"><code>code/engine/README.md</code></a>；它复用 Chapter06 Request/Runtime contract，并通过 Factory 把 22 层 CPU Model 接入 Chapter10 HTTP Adapter 和请求级 Worker。

## 对应文档

- [Chapter 06 导读](../chapter06_generation/next/intro.md)
- [Chapter 06 真实 TinyLlama 生成](../chapter06_generation/next/08_real_tinyllama_generation.md)
- [Chapter 10 服务代码导航](./chapter10_serving.md)
