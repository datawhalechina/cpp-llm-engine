# vLLM 高吞吐推理引擎

## 为什么读

[vLLM](https://github.com/vllm-project/vllm) 适合观察模型执行、请求调度和 KV 内存管理怎样组成高吞吐引擎。阅读重点不是某个 kernel，而是 EngineCore、scheduler、worker/model runner 和 cache manager 之间的职责。

## 重点看什么

- Continuous Batching 怎样按 iteration 重组运行请求
- Block/Paged KV 怎样把逻辑 token 位置映射到物理缓存
- Prefill 与 Decode 怎样共享执行器又保持不同 workload
- Prefix caching、抢占和恢复需要增加哪些状态与资源约束

本教程用 Chapter06 固定单请求推进语义，Chapter07-08 固定 K/V 与调度契约，Chapter10 固定服务生命周期，再用这些对象理解 vLLM 的多进程和 GPU 执行路径。

## 源码坐标

| 对象 | 官方入口 | 阅读问题 |
| --- | --- | --- |
| Llama model | [`llama.py`](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/llama.py) | DecoderLayer 怎样装配 Attention、MLP 和权重 |
| Model Loader | [`model_loader`](https://github.com/vllm-project/vllm/tree/main/vllm/model_executor/model_loader) | checkpoint 格式怎样映射到模型权重 |
| Attention | [`vllm/attention`](https://github.com/vllm-project/vllm/tree/main/vllm/attention) | model layer 怎样把 metadata 交给 backend |
| Quantization | [`quantization`](https://github.com/vllm-project/vllm/tree/main/vllm/model_executor/layers/quantization) | quant method 怎样绑定到模型 Linear |
| Scheduler | [`scheduler.py`](https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/sched/scheduler.py) | waiting/running 请求怎样组成执行计划 |
| KV cache manager | [`kv_cache_manager.py`](https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/kv_cache_manager.py) | block 何时分配、提交和释放 |
| EngineCore | [`engine/core.py`](https://github.com/vllm-project/vllm/blob/main/vllm/v1/engine/core.py) | scheduler、executor 和输出怎样协作 |
| Benchmark | [`benchmarks`](https://github.com/vllm-project/vllm/tree/main/benchmarks) | online、offline 和 latency workload 怎样区分 |

## 对应教程章节

- Chapter04：模型层、Attention backend 和权重映射
- Chapter06：Request、Prefill/Decode 和单请求 Runtime
- Chapter07：Block/Paged K/V 与缓存资源
- Chapter08：Scheduler、Continuous Batching 和 Tensor batch
- Chapter09：服务 workload、吞吐/延迟指标和 Profiling
- Chapter10：EngineCore、worker 与服务入口的分工
