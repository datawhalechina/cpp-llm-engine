# Repository Engine First Success 首个纵向切片

`code/engine` 把本教程自己的 CPU-only 纵向路径串成一个可运行切片，不替代 Chapter 00 的 llama.cpp 真实模型 baseline，也不是完整推理服务。

## 资产总览

| 资产 | 作用 |
| --- | --- |
| `include/` | toy Engine、真实 TinyLlama Engine、Tokenizer、执行器工厂和可选 CUDA 权重对象接口 |
| `src/` | prompt、Tokenizer backend、教学 decoder、真实 Model/Context 适配与 CUDA 单层权重上传 |
| `tests/` | core contract、engine、embedding Runtime 与 ServiceCore 集成测试 |
| `EmbeddingRequestExecutor` | 预计算 text/visual embedding -> Chapter06 每请求执行器 |
| `examples/` | First Success、真实生成/服务/HTTP smoke 和真实服务 benchmark |
| `scripts/run.sh` | 默认隔离构建、demo 和 CTest 入口 |
| `scripts/run_real_serving_benchmark.sh` | 可选 TinyLlama 双请求 Release benchmark |
| `scripts/run_real_batch_benchmark.sh` | 可选 TinyLlama sequential/scalar Linear batch 配对 benchmark，并输出 Chapter08 可消费的分段计时 |
| `scripts/run_real_decoder_profile.sh` | 可选 TinyLlama Decoder 子阶段计时与结果归档 |
| `scripts/run_real_http_server.sh` | 启动真实 TinyLlama HTTP server，或以 `--check` 执行 curl 闭环 |
| `CMakeLists.txt` | CPU 集成构建，以及显式启用的 CUDA 权重适配测试 |

## 运行与验证

```bash
cmake -S code/engine -B code/engine/build
cmake --build code/engine/build
code/engine/build/cpp_llm_engine_demo --prompt "hello" --max-new-tokens 4
ctest --test-dir code/engine/build --output-on-failure
```

默认构建使用 toy config、toy tokenizer 和教学 decoder，适合快速回归接口与状态机。可选真实路径读取 TinyLlama GGUF，执行真实 tokenizer、22 层 CPU Decoder、greedy sampling 和 detokenize；真实服务 smoke 使用 3-token step budget 验证 append-prefill、双请求隔离和 Block KV 回收，但不代表生产并发性能。

启用 CUDA 后，Engine 会用 `CudaLlamaLayerWeights` 把 Chapter01 的单层 host 权重转换为 Decoder 使用的 GPU 只读权重视图。默认模型和 CUDA Toolkit 位于教程推荐路径时，只需运行：

```bash
bash code/engine/scripts/run_real_cuda_first_layer.sh
```

模型或 Toolkit 位于其他位置时，通过 `CPP_LLM_ENGINE_MODEL_PATH` 和 `CUDA_HOME` 覆盖；需要保留构建目录排查问题时设置 `CUDA_REAL_LAYER_BUILD_DIR`。该入口需要真实 NVIDIA 设备，当前只执行单 token、单层；22 层 CUDA Model/Context 和 Generation Runtime 尚未接入。

`EmbeddingRequestExecutor` 接收 Core `DecoderEmbeddingSequence` 和每请求 `DecoderExecutionContext`，把预计算 embedding 适配为 Chapter06 `RequestModelExecutor`。Request 中的 prompt position 只承担序列长度和调度计数，真实 hidden values 由 Context 消费；prefill 成功后适配器立即释放 embedding storage，decode 继续使用同一 Context 和既有 KV 生命周期。默认测试用合成 Context 验证 Runtime 生成、长度校验、错误传播与释放，不要求多模态模型文件。

`embedding_request_service_test` 将同一个适配器注入 Chapter10 `HttpAdapter`。同步 completion 与真实 `ResponseWriter` SSE 都生成 `[2,4]`；压力段连续执行 32 个成功请求、8 个 prefill 失败请求，再覆盖客户端断开和 deadline 超时。每次终态后都检查 Context 析构、inflight 归零以及 accepted 与 completed/failed/cancelled/timed_out/released 计数。该测试固定服务复用契约，不解析图片 HTTP body，也不代替真实 SmolVLM 服务 benchmark。

Engine 已支持通过 `TokenizerBackend` 注入 tokenizer。默认路径使用 toy tokenizer；启用 llama.cpp 后，可验证真实 TinyLlama tokenizer 已进入 Engine 的 prompt、Request 和 Runtime 链路：

```bash
cmake -S code/engine -B /tmp/cpp-llm-engine-real-tokenizer \
  -DCHAPTER01_ENABLE_LLAMA_CPP=ON \
  -DLLAMA_CPP_ROOT="$PWD/llama.cpp"
cmake --build /tmp/cpp-llm-engine-real-tokenizer --target real_tokenizer_engine_smoke
/tmp/cpp-llm-engine-real-tokenizer/real_tokenizer_engine_smoke \
  llama.cpp/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
```

这个 smoke 只验证 tokenizer contract。继续构建真实生成目标，可验证同一套 Chapter06 Runtime 驱动真实模型：

```bash
cmake -S code/engine -B /tmp/cpp-llm-engine-real-generation \
  -DCHAPTER01_ENABLE_LLAMA_CPP=ON \
  -DLLAMA_CPP_ROOT="$PWD/llama.cpp" \
  -DCHAPTER05_SOURCE_DIR="$PWD/code/chapter05_quantization" \
  -DCPP_LLM_ENGINE_REAL_MODEL="$PWD/llama.cpp/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf" \
  -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/cpp-llm-engine-real-generation \
  --target real_tinyllama_generation_smoke real_tinyllama_batch_smoke \
  real_tinyllama_serving_smoke \
  real_tinyllama_http_adapter_smoke -j2
/tmp/cpp-llm-engine-real-generation/real_tinyllama_generation_smoke \
  llama.cpp/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf \
  4 "Hello TinyLlama"
ctest --test-dir /tmp/cpp-llm-engine-real-generation \
  -R 'real_tinyllama_(generation|batch|serving|http_adapter)_test' --output-on-failure
```

设置 `CPP_LLM_ENGINE_REAL_MODEL` 后才会注册标记为 `real_model;slow` 的真实模型 CTest，默认无模型构建仍只运行轻量测试。当前基线输出 6 个 prompt token，并生成 `29892 306 29915 29885`，解码文本为 `, I'm`。CTest 在同一 Engine 上连续生成两次，断言 `golden_tokens_match: true` 和 `context_isolation_match: true`。

`RealLlamaExecutorFactory` 长期持有只读 `RealLlamaCpuModel`，每次 `create_executor()` 创建一个独立 `RealLlamaCpuContext`。Context 独占 position 和 22 层连续 K/V。`decode_batch()` 校验 position 后，在每层合并多个 Context 的 Q/K/V、输出投影、FFN 和 LM Head Linear；scalar Q4_K/Q6_K/Q8_0 backend 一次遍历权重并累加多个输入行。真实 batch CTest 证明不同长度 prompt 的 logits 和固定输出与顺序执行一致，错误 position 或 token 不会提交状态。AVX2 batch 仍使用逐行兼容路径，KV 也未写入分页 payload，因此尚不宣称批量吞吐收益。

真实服务 smoke 用同一 Factory 为 `Hello TinyLlama` 和 `Hi` 创建两个 Executor；长 prompt 分块追加到同一 Context，中间 chunk 不产生 logits，固定输出仍分别为 `29892 306 29915 29885` 与 `839 744 29892 263`。terminal release 后，两个 Executor/Context 和 block table 均被释放。服务调度仍是单线程 step 正确性路径，不宣称线程安全。

`real_tinyllama_http_adapter_smoke` 把同一 Factory 注入 Chapter10 `HttpAdapter`，验证同步 JSON 返回第一组 token、SSE 按顺序返回第二组 token，并核对两个请求的 accepted/completed/released/generated metrics。需要继续验证真实 socket 时运行 `bash code/engine/scripts/run_real_http_server.sh --check`；去掉 `--check` 会保留服务供手工 curl。transport 只处理 HTTP/SSE，并用互斥锁串行调用当前单线程 ServiceCore。

真实服务性能基线使用独立脚本，默认执行一次 warmup 和三轮测量：

```bash
bash code/engine/scripts/run_real_serving_benchmark.sh
```

结果写入 `code/chapter09_performance/results/tinyllama_service_cpu_baseline.json`。`memory_bytes` 分别保存共享模型的 Tensor payload、两个 Context 在释放前实际提交的连续 K/V 和进程峰值 RSS；三者不是可直接相加的互斥分类。比较前必须保持 CPU、编译器、Release 构建、模型、prompt 和 token budget 一致；正确性字段失败时，不使用延迟、吞吐或内存数字做优化结论。

真实 scalar Linear batch 使用另一个配对实验：

```bash
bash code/engine/scripts/run_real_batch_benchmark.sh
```

该实验排除模型加载和 prefill，只比较两个真实 Context 的顺序 decode 与 batch decode。当前归档结果的 batch 均值比顺序路径更慢，因此该实现保留为正确性台阶，不作为默认性能路径。

需要继续定位真实 CPU Decoder 内部耗时时，运行：

```bash
bash code/engine/scripts/run_real_decoder_profile.sh
```

该脚本只在专用 Context 上开启 profiling，默认生成路径不读取时钟。结果写入 Chapter09 的历史性能资产目录；只有 golden token、Decoder 顶层计时和 FFN 子项闭合全部通过，脚本才接受并归档结果。

profile CLI 的第三个参数可以选择 `scalar`、`q4-avx2`、`q6-avx2`、`q4-attention-avx2` 或 `optimized-avx2`。前三个实验模式分别只切换 FFN gate/up、FFN down 或 Attention QKV/out，组合模式同时启用三个已验证作用域并保持 LM Head scalar；历史 A/B 脚本归档在 `code/chapter09_performance/legacy/chapter08_profiling/scripts/`。

AVX2 路径会在选择时检查 AVX2 与 FMA，不能运行时直接返回错误，不静默回退。
