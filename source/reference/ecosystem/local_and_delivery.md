# 本地推理与产品交付

llama.cpp 提供本地推理运行时和 GGUF 基线，Ollama 在推理能力之上组织模型管理与本地 API。二者分别用于对照引擎执行和产品交付。

<a id="llama-cpp"></a>
## llama.cpp

### 阅读目的

[llama.cpp](https://github.com/ggml-org/llama.cpp) 用较小的本地运行时连接 GGUF、CPU/GPU 后端、模型上下文和服务入口。本教程用它固定真实模型结果，并核对加载、量化、生成和本地服务策略。

### 重点对象

| 对象 | 官方入口 | 阅读问题 |
| --- | --- | --- |
| 模型与上下文 API | [`include/llama.h`](https://github.com/ggml-org/llama.cpp/blob/master/include/llama.h) | 模型权重和请求状态怎样分开 |
| 模型加载器 | [`llama-model-loader.cpp`](https://github.com/ggml-org/llama.cpp/blob/master/src/llama-model-loader.cpp) | GGUF 元数据、张量索引和数据区怎样校验 |
| CPU/CUDA 后端 | [`ggml`](https://github.com/ggml-org/llama.cpp/tree/master/ggml) | 同一算子怎样进入 CPU 计算核或 CUDA 实现 |
| 量化实现 | [`ggml-quants.c`](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-quants.c) | 分块格式怎样进入点积和矩阵乘法 |
| 性能基准 | [`llama-bench`](https://github.com/ggml-org/llama.cpp/tree/master/tools/llama-bench) | 预填充和生成怎样分别计时 |
| 本地服务 | [`tools/server`](https://github.com/ggml-org/llama.cpp/tree/master/tools/server) | 模型上下文怎样接入 HTTP 和请求槽位 |

### 对应章节

Chapter00–01 使用它建立外部基线和加载对照；Chapter02–06 核对张量、算子、量化与生成；Chapter09–10 对照性能工具和本地服务。

<a id="ollama"></a>
## Ollama

### 阅读目的

[Ollama](https://github.com/ollama/ollama) 主要展示模型管理、本地 API 和用户交付怎样包装底层推理能力。阅读时关注产品入口与运行时的交接，而不是用它替代算子和引擎实现。

### 重点对象

| 对象 | 官方入口 | 阅读问题 |
| --- | --- | --- |
| 项目入口 | [Ollama 仓库](https://github.com/ollama/ollama) | 模型、运行参数和本地服务怎样组成统一工作流 |
| API | [API 文档](https://github.com/ollama/ollama/blob/main/docs/api.md) | 生成请求、流式响应和模型管理怎样暴露给用户 |

### 对应章节

Chapter00 对照开箱即用的真实推理体验，Chapter06 对照生成接口，Chapter10 对照本地服务和流式响应。
