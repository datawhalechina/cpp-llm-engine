# Repository Engine First Success 首个纵向切片

`code/engine` 把本教程的 CPU 主线和可选 CUDA 后端串成可运行纵向切片，不替代 Chapter 00 的 llama.cpp 真实模型基线，也不是生产级推理服务。

## 对象与所有权

| 概念 | 代码角色 | 生命周期与所有权 |
| --- | --- | --- |
| 模型（Model） | 真实 CPU/CUDA Model | 长期复用，保存配置和共享只读权重 |
| 执行器工厂（Factory） | `RealLlamaExecutorFactory`、`RealLlamaCudaExecutorFactory` | 长期复用，共享 Model、块池或批量工作区，并创建请求执行器 |
| 请求执行器（Executor） | `RequestModelExecutor` 实现 | 每个请求独立，把 Prefill/Decode 转交给 Model 和 Context |
| 模型执行上下文（Context） | CPU/CUDA Context | 每个请求独立，保存或绑定位置、分层 K/V 和后端状态 |
| 批量工作区（Batch Workspace） | CUDA batch workspace | Factory 共享，只保存可复用临时张量，不保存请求历史 |

Factory 的名称表示它负责按统一规则创建和装配 Executor，而不是执行调度。Context 在这里特指请求运行状态对象，不等同于模型的上下文窗口容量。

## 启动参数与内部配置

可执行程序负责把命令行和环境默认值转换为类型化配置，Engine 内部不解析 `argv`。统一数据流为 `CLI -> Options/枚举 -> Factory -> Executor/Context -> Backend`。构建期开关继续由 CMake 管理，请求级生成参数进入 `GenerationConfig`，benchmark 的 batch 和迭代次数只属于 workload。

实现选择应满足三项条件：候选仍受维护、已通过正确性门禁、存在清晰的类型化枚举。未知值必须在加载模型前失败，程序应打印最终生效配置。实验编号、结果目录和优化名称只用于登记证据，不得隐式切换实现。

## 资产总览

| 资产 | 作用 |
| --- | --- |
| `include/` | toy Engine、真实 Llama/Qwen3 CPU Engine、Tokenizer 和执行器工厂接口 |
| `src/` | prompt、Tokenizer backend、教学 decoder 与真实 CPU Model/Context 适配 |
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

启用 CUDA 后，`code/model` 会把 Chapter01 的 host 权重上传为 Model 共享的 GPU 只读权重，并为请求创建独立 Context。默认模型和 CUDA Toolkit 位于教程推荐路径时，只需运行：

```bash
bash code/engine/scripts/run_real_cuda_first_layer.sh
```

模型或 Toolkit 位于其他位置时，通过 `CPP_LLM_ENGINE_MODEL_PATH` 和 `CUDA_HOME` 覆盖；需要保留构建目录排查问题时设置 `CUDA_REAL_LAYER_BUILD_DIR`。该入口快速验收单 token、单层。继续运行下面的命令，会用同一 embedding 核对 CPU/CUDA 的 2 层与完整 22 层 hidden 和分层 K/V：

```bash
bash code/engine/scripts/run_real_cuda_decoder.sh
```

完整 Decoder 对照用于定位 hidden 与分层 K/V。继续运行 Runtime 入口，会从固定 token prompt 开始执行 CUDA embedding、22 层 Decoder、final norm、LM Head 和 greedy decode：

```bash
bash code/engine/scripts/run_real_cuda_runtime.sh
```

Qwen3 使用相同 Model、Factory、Executor 和 Runtime 接口，模型元数据会自动选择 28 层、Q/K Norm、NeoX RoPE 和非对称 Q/KV 投影。按首层、全层、Runtime 的顺序运行，便于把结构错误定位在最小阶段：

```bash
bash code/engine/scripts/run_qwen3_cuda_first_layer.sh
bash code/engine/scripts/run_qwen3_cuda_decoder.sh
bash code/engine/scripts/run_qwen3_cuda_runtime.sh
bash code/engine/scripts/run_qwen3_cuda_generation.sh
bash code/engine/scripts/run_qwen3_cuda_serving.sh
```

Qwen3 Runtime 入口同时检查确定性生成、双请求 Tensor Batch、持久化工作区、分页 K/V 和 Radix 前缀复用。真实文本入口继续接入 llama.cpp Tokenizer，并用 CPU 路径已经登记的 `11 358 1366 311` 核对 CUDA 输出。服务入口复用同一 Factory，并已通过 Completion、SSE、取消、超时、批量调度和终态资源归零验收。

继续运行真实文本入口，会使用与 CPU Engine 相同的 llama.cpp tokenizer，并核对四个生成 token：

```bash
bash code/engine/scripts/run_real_cuda_generation.sh
```

最后把同一个执行器工厂注入 Chapter10，验证同步 Completion、SSE、取消、超时和终态资源释放：

```bash
bash code/engine/scripts/run_real_cuda_serving.sh
```

当前 CUDA 执行器工厂已实现 Chapter08 `ExecutorBatchRunner`。Decode batch 把多个请求上下文的 hidden 打包为批量 Tensor，共享 RMSNorm、QKV/输出投影、FFN 和 LM Head；RoPE 使用每请求位置，Attention 根据 Context 访问连续 K/V 或各自页表映射的分页 payload。整批失败会恢复所有 Context，错误 position、请求绑定或未预留槽位在执行前拒绝。

连续和分页 CUDA Executor 的完整 prompt 都使用 `RealLlamaCudaContext::prefill_tokens()`，在请求级持久 Prefill workspace 中执行批量 embedding、多层 Tensor Prefill 和最终 logits；后续生成仍逐 token Decode。分页路径先按 BlockTable 预留整个 chunk，逐层写入共享设备 payload，并用多 Query 因果 Attention 直接读取离散物理块；完整 chunk 成功后才提交位置，失败则裁剪回调用前状态。

`RealLlamaCudaExecutorFactory::set_attention_mode()` 可在创建请求执行器前选择三阶段或融合在线 Attention。默认值保持三阶段；Chapter09 的 EXP-GPU-04 表明融合实现已通过端到端正确性，但两轮真实 Prefill/Decode 中位数没有稳定收益，因此该开关用于教学 A/B 和继续调优，不代表生产默认策略。

Chapter09 的真实 Attention 入口将命令行 `--attention-path three-stage|fused-online` 转换为 `CudaAttentionMode`；量化 GEMV 实验同样把 `--gemv-path fixed-four|adaptive` 转换为 `CudaQuantizedGemvMode`。Factory 把类型化策略下发给请求 Context 和持久化 batch workspace，字符串不会进入 Model 或 kernel。Attention 的 `compare` 会在同一进程中核对两条路径；GEMV 使用独立进程和交错执行顺序，避免时钟与热状态固定偏向其中一条路径。

`real_tinyllama_cuda_adaptive_gemv_matrix` 复用同一个 Factory 和模型权重，按单元重新创建请求 Context。Prefill 后先通过 Chapter07 事务接口为完整 Decode 区间预留分页块，再开始计时；每次批量 Decode 成功后统一提交控制面的 token count，然后进入下一步。这与 Chapter08 调度器“先 reserve、执行成功后 commit”的契约一致，也能覆盖上下文恰好填满物理块的边界。多步 Decode 持续更新 token、position 和分页 K/V，最后比较顺序执行与 Tensor Batch 的 logits。命令行只定义 workload，路径仍通过 `CudaQuantizedGemvMode` 下发。

EXP-GPU-16 使用该入口覆盖 batch 1/2、上下文 `16/128` 和 Decode `1/8` 步。Adaptive 在 batch 1 的 Tensor Batch 延迟降低约 `18.4%`～`23.1%`，在 batch 2 降低约 `12.9%`～`14.7%`；两轮反向执行顺序下全部 logits 最大误差为 `0`。该结果只支持当前小 batch、Q4_K TinyLlama 和 RTX 5070 Ti Laptop 环境，不直接外推其他模型或设备。

执行器工厂按精确 batch size 缓存 `RealLlamaCudaBatchWorkspace`，复用 stream、cuBLAS handle、embedding rows、批量 hidden/logits 和 Decoder 中间 Tensor。同一工厂的批量调用由互斥锁串行保护；请求级位置和页表仍归各模型执行上下文所有，不进入 workspace。真实 Runtime 与 Service 测试覆盖双请求分页 batch、预留/提交分离、失败不推进和终态释放。融合 PagedAttention、跨 batch size 容量复用、并发 batch 和真实 socket server 尚未接入。

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

`RealLlamaExecutorFactory` 长期持有只读 CPU Model，每次 `create_executor()` 创建一个独立 Context。Model 从 GGUF 自动识别 `llama` 或 `qwen3`；Context 独占 position 和每层连续或分页 K/V。`decode_batch()` 校验 position 后，在每层合并多个 Context 的 Q/K/V、输出投影、FFN 和 LM Head Linear；scalar Q4_K/Q6_K/Q8_0 backend 一次遍历权重并累加多个输入行。真实 batch 测试证明不同长度 prompt 的 logits 和固定输出与顺序执行一致，错误 position 或 token 不会提交状态。AVX2 batch 仍使用逐行兼容路径，因此尚不宣称批量吞吐收益。

启用分页 K/V 后，CPU 和 CUDA Factory 都持有 `RadixPrefixCache`。请求结束时，完整物理块在释放请求页表前发布到前缀索引；新请求准入只估算可复用长度，获得运行槽时才重新查询并同步共享 `BlockTable`、Request 的 Prefill 进度和 CPU/CUDA Context 位置。`PrefixBlockStore` 让索引共用匹配和淘汰算法，同时保留两种 payload 的清理时序。容量不足时，数据面先回收最久未使用的索引独占叶端块，再重试预留；真实 CPU 和 CUDA 测试都使用相同四词元前缀比较冷启动与缓存命中路径，验证 logits 或首个生成词元一致。

真实服务 smoke 根据 Factory 报告的架构选择 TinyLlama 或 Qwen3 固定 prompt 与 golden，并为两个请求创建独立 Executor。长 prompt 分块追加到同一 Context，中间 chunk 不产生 logits；测试还会先发布一组真实 K/V 前缀，再让冷请求与命中请求依次入队，检查后者先执行。terminal release 后 Executor/Context 和请求 block table 被释放，Radix 索引可继续持有可复用物理块。HTTP 组合测试用同一配置验证同步 Completion、SSE 和 metrics；`real_serving_match: true` 是 ServiceCore 闭环的总通过标记。

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
