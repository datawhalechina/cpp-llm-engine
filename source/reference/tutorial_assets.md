# 教程资产索引

教程资产索引回答两个问题：每章从哪里进入，关键对象和算法在哪里实现。单章阅读顺序以章节导读和代码 README 为准；实验编号、原始结果与复现脚本单独登记在[实验资产索引](./experiment_assets.md)。

## 按章节查找

| 章节 | 文档入口 | 代码入口 | 最小验证 |
| --- | --- | --- | --- |
| Chapter00 真实基线 | [导读](../chapter00_quickstart/intro.md) | [代码资产](/code/chapter00_quickstart) | `bash code/chapter00_quickstart/run.sh` |
| Chapter01 输入与权重 | [导读](../chapter01_tensor_loader/intro.md) | [代码资产](/code/chapter01_tensor_loader) | `bash code/chapter01_tensor_loader/run.sh` |
| Chapter02 Tensor 与 View | [导读](../chapter02_tensor_ops/intro.md) | [代码资产](/code/chapter02_tensor_ops) | `bash code/chapter02_tensor_ops/scripts/run.sh` |
| Chapter03 Operator 与 Backend | [导读](../chapter03_matmul/intro.md) | [代码资产](/code/chapter03_matmul) | `bash code/chapter03_matmul/scripts/run.sh` |
| Chapter04 Decoder | [导读](../chapter04_attention/intro.md) | [代码资产](/code/chapter04_attention) | `bash code/chapter04_attention/scripts/run.sh` |
| Chapter05 Quantization | [导读](../chapter05_quantization/intro.md) | [代码资产](/code/chapter05_quantization) | `bash code/chapter05_quantization/scripts/run.sh` |
| Chapter06 Generation | [导读](../chapter06_generation/next/intro.md) | [代码资产](/code/chapter06_generation) | `bash code/chapter06_generation/scripts/run.sh` |
| Chapter07 K/V Cache | [导读](../chapter07_kv_cache/next/intro.md) | [代码资产](/code/chapter07_kv_cache) | `bash code/chapter07_kv_cache/scripts/run.sh` |
| Chapter08 Scheduling | [导读](../chapter08_scheduling/next/intro.md) | [代码资产](/code/chapter08_scheduling) | `bash code/chapter08_scheduling/scripts/run.sh` |
| Chapter09 Performance | [导读](../chapter09_performance/next/intro.md) | [代码资产](/code/chapter09_performance) | `bash code/chapter09_performance/scripts/run.sh` |
| Chapter10 Serving | [导读](../chapter10_serving/next/intro.md) | [代码资产](/code/chapter10_serving) | `bash code/chapter10_serving/scripts/run.sh` |

真实模型组装、CUDA 和多模态分别从 [`code/engine`](/code/engine)、[`code/cuda`](/code/cuda) 和 [`code/multimodal`](/code/multimodal) 进入。

## 核心数据结构

| 对象 | 解决的问题 | 首次出现 | 代码入口 | 当前证据 |
| --- | --- | --- | --- | --- |
| `TokenSequence`、`TokenizerBackend` | 在文本、词元 ID 和嵌入之间传递输入 | Chapter01 | <a href="https://github.com/datawhalechina/cpp-llm-engine/blob/main/code/engine/include/cpp_llm_engine/tokenizer_backend.h"><code>tokenizer_backend.h</code></a> | 真实分词器已接入推理引擎 |
| GGUF 元数据、`ModelConfig` | 保存模型架构、维度和量化布局 | Chapter01 | <a href="https://github.com/datawhalechina/cpp-llm-engine/blob/main/code/chapter01_tensor_loader/src/gguf_reader.cpp"><code>gguf_reader.cpp</code></a> | 字段模式与真实模型检查 |
| `Storage`、shape、stride、offset | 表达连续张量和共享存储视图 | Chapter02 | <a href="https://github.com/datawhalechina/cpp-llm-engine/blob/main/code/chapter02_tensor_ops/include/chapter02_tensor_ops/tensor.h"><code>tensor.h</code></a> | reshape、slice 与越界契约 |
| `CudaTensor`、`CudaExecutionContext` | 绑定设备存储并复用 stream、handle 和工作区 | [CPU/GPU 架构](./architecture.md#cpu-主线与-gpu-扩展) | <a href="https://github.com/datawhalechina/cpp-llm-engine/blob/main/code/cuda/include/cpp_llm_engine/cuda/cuda_execution_context.h"><code>cuda_execution_context.h</code></a> | 真实 CUDA 资源与数值测试 |
| `DecoderModelContract` | 描述教学层和真实模型共享的解码器结构 | Chapter04 | <a href="https://github.com/datawhalechina/cpp-llm-engine/blob/main/code/core/include/cpp_llm_engine/core/decoder_contract.h"><code>decoder_contract.h</code></a> | 教学层与 TinyLlama 22 层校验 |
| `DecoderExecutionContext` | 隔离请求位置、分层 K/V 和执行状态 | Chapter06 | <a href="https://github.com/datawhalechina/cpp-llm-engine/blob/main/code/model/include/cpp_llm_engine/model/real_llama_cpu_decoder.h"><code>real_llama_cpu_decoder.h</code></a> | 双上下文、生成与服务测试 |
| `Request`、`TokenHistory` | 保存生成状态和已提交词元 | Chapter06 | <a href="https://github.com/datawhalechina/cpp-llm-engine/blob/main/code/chapter06_generation/include/chapter06_generation/request.h"><code>request.h</code></a> | 状态迁移、追加和隔离测试 |
| `RequestModelExecutor`、Factory | 为请求创建上下文并适配模型调用 | Chapter06 | <a href="https://github.com/datawhalechina/cpp-llm-engine/blob/main/code/chapter06_generation/include/chapter06_generation/model_executor.h"><code>model_executor.h</code></a> | 分块预填充与真实模型服务 |
| `KvBlockPool`、`BlockTable` | 管理物理块并映射逻辑位置 | Chapter07 | <a href="https://github.com/datawhalechina/cpp-llm-engine/blob/main/code/chapter07_kv_cache/include/chapter07_kv_cache/block_table.h"><code>block_table.h</code></a> | 预留、回滚、释放和复用 |
| `CpuPagedKvCache`、`CudaPagedKvCache` | 保存分页 K/V 并按页表读取 | Chapter07 | <a href="https://github.com/datawhalechina/cpp-llm-engine/blob/main/code/chapter07_kv_cache/include/chapter07_kv_cache/paged_kv_cache.h"><code>paged_kv_cache.h</code></a> | CPU/CUDA 单请求与批次验证 |
| `StepBatchPlan` | 把等待和运行请求组织为单步计划 | Chapter08 | <a href="https://github.com/datawhalechina/cpp-llm-engine/blob/main/code/chapter08_scheduling/include/chapter08_scheduling/step_scheduler.h"><code>step_scheduler.h</code></a> | 预算、公平性与失败回滚 |
| `EvidenceRecord` | 保存工作负载、环境、样本和统计 | Chapter09 | <a href="https://github.com/datawhalechina/cpp-llm-engine/blob/main/code/chapter09_performance/include/chapter09_performance/evidence_record.h"><code>evidence_record.h</code></a> | 版本化 JSON 与 A/B 决策 |
| `ResponseWriter`、`ServiceMetrics` | 输出流式事件并记录服务终态 | Chapter10 | <a href="https://github.com/datawhalechina/cpp-llm-engine/blob/main/code/chapter10_serving/include/chapter10_serving/response_writer.h"><code>response_writer.h</code></a> | HTTP/SSE、取消和资源释放 |

## 关键算法

| 算法 | 解决的问题 | 首次出现 | 实现入口 | 当前证据 |
| --- | --- | --- | --- | --- |
| 分词器 encode/decode | 在文本和词元序列之间转换 | Chapter00–01 | llama.cpp 分词器适配器 | 固定提示词对照 |
| reshape、slice、步长寻址 | 在共享存储上改变观察方式 | Chapter02 | `Tensor::try_reshape()`、`try_slice()` | shape、offset 与别名测试 |
| MatMul、cuBLAS SGEMM | 完成线性层核心乘加 | Chapter03 | `MatMulOp`、`CublasMatmulBackend` | CPU/CUDA 多形状数值对照 |
| RMSNorm、RoPE、GQA、SwiGLU | 组成 Llama 解码器层 | Chapter04 | Chapter04 与 `code/model` | 教学结果和 22 层真实模型 |
| Q4_K/Q6_K/Q8_0 解码与量化 Linear | 直接消费压缩权重 | Chapter05 | `QuantizedLinearPolicy`、CUDA GEMV | 参考、SIMD、CUDA 与真实生成 |
| Prefill、Decode、采样 | 从提示词推进到生成终态 | Chapter06 | `Runtime`、`Sampler` | 单请求、流式和真实 TinyLlama |
| Paged K/V、Radix Cache | 分页保存历史并复用共享前缀 | Chapter07 | `CpuPagedKvCache`、`RadixPrefixCache` | 页表、引用计数和采用事务 |
| Continuous Batching | 按词元和缓存预算重组请求 | Chapter08 | `StepScheduler`、批执行器 | 混合步骤、防饥饿和真实批次 |
| 性能基准、剖析与 A/B | 判断优化是否有效并解释原因 | Chapter09 | Chapter09 脚本和结果文件 | [实验资产索引](./experiment_assets.md) |
| 准入、背压和终态释放 | 持续服务多个并发请求 | Chapter10 | `ServiceCore`、`RequestWorker` | 同步、SSE、取消与超时 |

## 阅读顺序

先用 `Storage → Tensor view → MatMul → Decoder` 建立模型计算主线，再用 `Request → Runtime → Paged K/V → Scheduler` 建立多请求主线，最后用 `EvidenceRecord → ServiceCore → ResponseWriter` 连接性能证据和在线服务。成熟框架中的对应对象按问题查阅[推理生态项目索引](./ecosystem/README.md)。
