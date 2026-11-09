# 生产级推理框架

vLLM、SGLang、TensorRT-LLM 和 DeepInfer 分别提供调度缓存、GPU 执行和大规模部署参照。阅读目标是理解教学对象在系统规模扩大后增加了哪些约束。

<a id="vllm"></a>
## vLLM

### 阅读目的

[vLLM](https://github.com/vllm-project/vllm) 展示模型执行、请求调度和 K/V 内存管理怎样组成高吞吐引擎。重点关注 EngineCore、调度器、模型执行器和缓存管理器的职责。

### 重点对象

| 对象 | 官方入口 | 阅读问题 |
| --- | --- | --- |
| 模型与加载 | [`model_executor`](https://github.com/vllm-project/vllm/tree/main/vllm/model_executor) | 模型层、权重和量化方法怎样装配 |
| 调度器 | [`scheduler.py`](https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/sched/scheduler.py) | 等待和运行请求怎样组成执行计划 |
| K/V 管理 | [`kv_cache_manager.py`](https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/kv_cache_manager.py) | 块何时分配、提交和释放 |
| EngineCore | [`engine/core.py`](https://github.com/vllm-project/vllm/blob/main/vllm/v1/engine/core.py) | 调度、执行和输出怎样协作 |
| 性能基准 | [`benchmarks`](https://github.com/vllm-project/vllm/tree/main/benchmarks) | 离线、在线和延迟负载怎样区分 |

### 对应章节

Chapter04–06 对照模型执行与生成，Chapter07–08 对照分页缓存和连续批处理，Chapter09–10 对照服务负载与 EngineCore。

<a id="sglang"></a>
## SGLang

### 阅读目的

[SGLang](https://github.com/sgl-project/sglang) 展示前缀 K/V 复用怎样影响请求排序、缓存生命周期和资源预算。LPM 是调度策略，Radix Cache 是缓存索引，HiCache 是多级存储扩展。

### 重点对象

| 对象 | 官方入口 | 阅读问题 |
| --- | --- | --- |
| 调度器 | [`scheduler.py`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/scheduler.py) | 词元与缓存预算怎样进入调度循环 |
| Radix Cache | [`radix_cache.py`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/mem_cache/radix_cache.py) | 前缀匹配、锁定、插入和淘汰怎样改变所有权 |
| SRT 运行时 | [`python/sglang/srt`](https://github.com/sgl-project/sglang/tree/main/python/sglang/srt) | API、调度器、执行器和缓存怎样分层 |
| 性能基准 | [`benchmark`](https://github.com/sgl-project/sglang/tree/main/benchmark) | 服务和缓存负载怎样定义 |

### 对应章节

Chapter06 对照请求推进，Chapter07 对照前缀缓存，Chapter08 对照缓存感知调度，Chapter09–10 对照服务指标和网络入口。

<a id="tensorrt-llm"></a>
## TensorRT-LLM

### 阅读目的

[TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) 展示 NVIDIA GPU 上的引擎构建、设备运行时、量化计算核和请求执行器怎样分层。

### 重点对象

| 对象 | 官方入口 | 阅读问题 |
| --- | --- | --- |
| C++ 运行时 | [`cpp/tensorrt_llm/runtime`](https://github.com/NVIDIA/TensorRT-LLM/tree/main/cpp/tensorrt_llm/runtime) | 引擎、缓冲区和执行上下文怎样组织 |
| GPU 计算核 | [`cpp/tensorrt_llm/kernels`](https://github.com/NVIDIA/TensorRT-LLM/tree/main/cpp/tensorrt_llm/kernels) | Attention、量化和融合算子在哪里实现 |
| 执行器 | [`cpp/tensorrt_llm/executor`](https://github.com/NVIDIA/TensorRT-LLM/tree/main/cpp/tensorrt_llm/executor) | 请求、批次和运行时怎样连接 |
| 性能基准 | [`benchmarks`](https://github.com/NVIDIA/TensorRT-LLM/tree/main/benchmarks) | 引擎配置和 GPU 条件怎样进入结果 |

### 对应章节

Chapter03–05 对照 GPU 算子、Attention 和量化，Chapter06–08 对照执行器与批次，Chapter09 对照设备性能证据。

<a id="deepinfer"></a>
## DeepInfer

### 阅读目的

[DeepInfer](https://github.com/deepseek-ai/DeepInfer) 展示 DeepSeek 模型在大规模服务中如何联合考虑模型结构、计算核、并行、异步调度和部署拓扑。掌握单机运行时后，可用它继续理解集群级实现。

### 重点对象

| 对象 | 官方入口 | 阅读问题 |
| --- | --- | --- |
| 项目实现 | [DeepInfer 仓库](https://github.com/deepseek-ai/DeepInfer) | Python 引擎与 C++ 调度器、块管理器怎样分工 |
| 阶段执行 | [DeepInfer 仓库](https://github.com/deepseek-ai/DeepInfer) | 视觉编码、预填充和解码分离后怎样管理资源与指标 |

### 对应章节

Chapter08 对照异步调度，Chapter09 对照分阶段负载，Chapter10 对照服务与集群基础设施；多模态路线对照视觉阶段分离。
