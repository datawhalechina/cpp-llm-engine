# llama.cpp 本地推理基线

## 解决什么问题

`llama.cpp` 是最小、最实用的本地推理 baseline。它回答的是这样一个问题：如何用尽量小的运行时开销，让模型在本地机器上先跑起来。

## 场景定位

- 适合本地 / 端侧先跑通第一条推理链路
- 适合看清楚一个通用本地运行时需要哪些最小组件
- 适合作为后续 chapter 的 baseline 参照

## 工程策略

- 保持运行时简单、可移植
- 聚焦本地部署和 GGUF 工作流
- 先保证正确性和可用性，再考虑大规模 serving
- 用最小可行的运行时，把模型文件和本地推理连接起来

## 源码坐标

| 对象 | 官方入口 | 阅读问题 |
| --- | --- | --- |
| 模型与 Context API | [`include/llama.h`](https://github.com/ggml-org/llama.cpp/blob/master/include/llama.h) | 模型权重和请求状态怎样分开 |
| Model Loader | [`llama-model-loader.cpp`](https://github.com/ggml-org/llama.cpp/blob/master/src/llama-model-loader.cpp) | GGUF metadata、tensor index 和 payload 怎样校验 |
| GGUF/ggml | [`ggml`](https://github.com/ggml-org/llama.cpp/tree/master/ggml) | Tensor、量化格式和 backend 怎样连接 |
| CPU Backend | [`ggml-cpu`](https://github.com/ggml-org/llama.cpp/tree/master/ggml/src/ggml-cpu) | graph op 怎样进入 CPU kernel |
| CUDA Backend | [`ggml-cuda`](https://github.com/ggml-org/llama.cpp/tree/master/ggml/src/ggml-cuda) | 同一 op 怎样选择 CUDA kernel 或库调用 |
| 量化实现 | [`ggml-quants.c`](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-quants.c) | block layout 怎样进入 CPU dot/matmul |
| Benchmark | [`llama-bench`](https://github.com/ggml-org/llama.cpp/tree/master/tools/llama-bench) | prompt processing 和 generation 怎样分别计时 |
| 本地服务 | [`tools/server`](https://github.com/ggml-org/llama.cpp/tree/master/tools/server) | C++ 模型上下文怎样接入 HTTP 和 slot |

## 为什么和 `Gemma` 很适合一起看

`llama.cpp` 更广、更通用，适合让读者看到很多本地推理系统最终都会共享的运行时形状；`Gemma` 则更具体，适合让读者看到某个模型家族是怎么落到 C++ 运行时里的。两者一起看，就同时有了平台视角和模型家族视角。

如果目标是理解一个通用的本地推理 baseline，就先看 `llama.cpp`；如果目标是理解一个具体模型家族如何映射到 C++ 运行时，就把它和 `Gemma` 对照着看。

## 对应教程章节

`llama.cpp` 主要对应教程里的这些章节：

- `chapter00_quickstart`
- `chapter01_tensor_loader`
- `chapter06_generation`
