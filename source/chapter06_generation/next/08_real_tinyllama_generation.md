# 06.8 验证真实模型单请求生成

合成 logits 可以稳定验证状态和停止条件，但不能证明 Tokenizer、完整 Decoder、量化权重和模型 K/V 已经连通。真实模型验证保持同一个 `Request -> Runtime -> Sampler` 控制流，只把教学执行器替换为真实模型 Context。TinyLlama 是默认基线，Qwen3 验证第二种稠密架构，Qwen1.5-MoE 验证标准专家模型，DeepSeek-V2-Lite 验证压缩 MLA 与混合稠密/MoE 层。

## 文本怎样进入单请求生成核心

真实入口由 `RealLlamaEngine` 组装，执行顺序如下：

| 顺序 | 对象 | 可观察结果 |
| --- | --- | --- |
| 1 | 真实 Tokenizer | prompt 文本变为当前模型的 token IDs |
| 2 | `RealLlamaExecutorFactory` | 从 GGUF 识别 `llama` 或 `qwen3`，加载一次只读权重并创建独立 Context |
| 3 | `Runtime::generate()` | 执行 Prefill、Greedy、Decode 和停止判断 |
| 4 | 真实 Tokenizer | 生成 token IDs 还原为文本 |

可执行入口位于 <a href="https://github.com/datawhalechina/cpp-llm-engine/blob/main/code/engine/examples/real_tinyllama_generation_smoke.cpp"><code>real_tinyllama_generation_smoke.cpp</code></a>。同一个 `RealLlamaEngine` 连续生成两次时会创建两份请求 Context，权重可以共享，位置和模型 K/V 不能共享。

## 固定模型与结果

固定模型、prompt 和 Greedy 后，CPU reference 保存两组 golden：

| 模型 | prompt | prompt token 数 | 生成 token IDs | 解码文本 |
| --- | --- | ---: | --- | --- |
| TinyLlama 1.1B | `Hello TinyLlama` | 6 | `29892 306 29915 29885` | `, I'm` |
| Qwen3 0.6B | `Hello Qwen3` | 4 | `11 358 1366 311` | `, I want to` |

两组结果都生成 4 个词元，并要求同一个 `RealLlamaEngine` 连续执行两次时结果一致。模型路径决定加载哪个 GGUF，架构由 `general.architecture` 自动选择；调用方不另外传入架构字符串。

Golden 用于发现 Loader、Tokenizer、量化 Linear、Attention 或 Runtime 的行为变化，不用于评价回答质量。模型版本、Tokenizer 或数值 backend 改变后，应先定位差异来源，再决定是否更新结果。

## MoE 怎样复用同一生成循环

Qwen1.5-MoE 仍把词元、logits 和停止状态交给 Chapter06。区别位于模型执行器：`RealQwen2MoeExecutorFactory` 识别 `qwen2moe` 架构，共享 24 层只读权重，并为每个请求创建独立 Context。Context 保存位置和每层 K/V；Prefill 与 Decode 进入模型后，再执行 Router、Top-K、命中专家和共享专家。

因此 Generation Runtime 无需理解专家数量或量化格式。它只调用统一的 Prefill/Decode 接口，模型 Context 负责维护内部状态；Chapter08 再把多个 Context 组织成批次。

DeepSeek-V2-Lite 使用相同边界。`RealDeepSeek2ExecutorFactory` 共享 GGUF 索引和全局权重，每个执行器独占 27 层 Context；Context 在每层保存 512 维 latent K/V 与 64 维位置 Key。Runtime 仍只观察 logits、已提交词元数和停止状态，不感知 MLA、64 选 6 路由或首层稠密 FFN。

CUDA Model 沿用相同的所有权划分。`RealDeepSeek2CudaModel` 常驻一层稠密权重、26 层 MoE 权重和输出权重；`RealDeepSeek2CudaContext` 独占执行流、中间 Tensor 与每层压缩历史。`RealDeepSeek2CudaExecutorFactory` 把这组对象适配为 Chapter06 的 Prefill/Decode 接口。单词元 CPU/CUDA logits 余弦相似度为 `1`，固定输入词元 `11` 的贪心结果同为 `207`。

分块 Prefill 接收一段词元后，先逐行查 embedding 并打包为 `[T,D]`，再让这个 Tensor 穿过 27 层。稠密层使用多 Query 因果 MLA 与批量 FFN；MoE 层的 Router 一次读取整批 hidden，设备端再按专家索引收集命中行，完成 Gate、Up 和 Down 后按路由权重散回原行。逐行拆分与重新打包只作为数值和性能对照路径保留。连续与分页 MLA 共用这条计算路径，只替换历史数据的地址解释。整个 chunk 成功后才推进位置，任一层失败都恢复 hidden 与缓存。这样验证的是一次多词元模型前向，而不是在 `prefill_chunk()` 内隐藏逐词元 Decode 循环。

整段 Tensor Prefill 的价值不只来自接口调用更少。逐词元基线会让 27 层权重、工作区准备和 Kernel 启动随词元数重复；Tensor 路径把 `[T,D]` 保持到层边界，在一次层前向中复用权重并并行处理多个 Query。[EXP-GPU-19](../../reference/experiment_assets.md#exp-gpu-19) 使用相同模型、连续 MLA 和最终 logits，对长度 `2/8/32/128` 做反向顺序配对。Tensor/逐词元中位数比例依次为 `0.552/0.258/0.183/0.343`，全部 logits 最大误差为 `0`。长度 128 的比例回升说明因果 Attention 随上下文增长后的占比已经上升，但 Tensor 路径仍明显快于逐词元调用。

模型权重缓存与请求 Context 是两个对象。DeepSeek Factory 默认逐层读取并释放权重，因此消费内存较低；将缓存容量设为 27 后，后续 Decode 可以复用全部层权重，但会增加接近整套 Decoder 的常驻内存。容量小于 27 的普通 LRU 对顺序扫描通常没有命中，因为同一层再次出现前已经访问了其余 26 层。无论是否缓存权重，请求位置和 MLA 历史始终保存在各自 Context 中。

Factory 同时提供跨请求 Decode 接口。它按层加载一次权重，每个请求先从自己的压缩 MLA 历史计算 Attention，再把同层 FFN 输入按专家分组执行，最后拆回各请求。整批成功后才提交 Context；任一层失败都会恢复批次开始前的状态。当前 CPU 路径已经实现专家分组，MLA、共享专家和 LM Head 仍按请求计算，融合 Tensor 执行留给后续优化。

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

准备好 Qwen3 固定 GGUF 后，运行第二种架构的相同验收：

```bash
bash code/chapter06_generation/scripts/run_qwen3_generation.sh
```

输出中的 `architecture: qwen3` 证明架构来自模型文件，golden 与请求隔离检查证明 Chapter06 控制流无需为 Qwen3 复制一套实现。

准备好 Qwen1.5-MoE 固定 GGUF 后，运行专家模型的生成测试：

```bash
bash code/engine/scripts/run_qwen15_moe_generation.sh
```

入口使用真实分词器执行 24 层 Prefill 和 Decode，默认生成词元 `11 358`，解码为 `, I`。输出 `qwen15_moe_text_generation: PASS` 表示文本、模型执行和请求状态已经连通。

准备好 DeepSeek-V2-Lite 固定 GGUF 后，运行 MLA-MoE 模型的生成测试：

```bash
bash code/engine/scripts/run_deepseek_v2_lite_generation.sh
```

默认输入 `Hello`，生成词元 `11 304`，解码为 `, I`。`golden_tokens_match: true` 核对固定结果，`deepseek2_text_generation: PASS` 表示 Tokenizer、27 层 Decoder、LM Head、Runtime 和请求 Context 已经连通。

有 NVIDIA GPU 时，先运行完整模型数值对照，再运行 Chapter06 Runtime：

```bash
CUDA_HOME=/usr/local/cuda-12.8 \
bash code/engine/scripts/run_deepseek_v2_lite_cuda_model.sh

CUDA_HOME=/usr/local/cuda-12.8 \
bash code/engine/scripts/run_deepseek_v2_lite_cuda_runtime.sh
```

第二个入口输出 `generated_token: 207`、`chunked_prefill_position: 2` 和 `context_isolation_match: true`。这组结果证明 CUDA Model 已进入单请求生成状态机；跨请求 Tensor Batch 和服务生命周期分别由 Chapter08、Chapter10 验证。
