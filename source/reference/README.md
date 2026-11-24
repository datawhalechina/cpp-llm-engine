# 参考资料索引

Reference 是按问题查阅的资料区，不是 Chapter00–10 之后必须顺序阅读的新章节。先确定要查的是概念、模型、系统结构、验证证据还是成熟框架，再进入对应页面。

## 每份资料只回答一类问题

| 文件 | 唯一职责 | 不在这里维护 |
| --- | --- | --- |
| [系统与执行架构](./architecture.md) | 本教程对象关系、CPU/GPU 边界和策略归属 | 模型完成度、实验数值、外部仓库源码坐标 |
| [模型能力](./model_support.md) | 各模型架构差异、接入层级和规划顺序 | 通用系统架构、完整运行命令、性能样本 |
| [公式索引](./formulas.md) | 稳定公式、符号、shape 和计算口径 | 章节推导过程、术语百科、实验结论 |
| [术语表](./glossary.md) | 中文名称、英文缩写和稳定定义 | 公式推导、项目版本、实现状态 |
| [实验资产](./experiment_assets.md) | 实验 ID、负载、事实来源和结论边界 | 教学原理、架构总览、重复原始样本 |
| [生态项目卡](./ecosystem/README.md) | 外部项目用途和可继续追踪的源码入口 | 本教程能力状态和章内原理讲解 |

章节正文负责第一次讲清原理和本章实现；代码 README 负责本地代码架构与运行入口。Reference 只提供跨章节查询，不复制这两类正文。

| 要解决的问题 | 首选入口 | 继续查阅 |
| --- | --- | --- |
| 一个术语、公式、数据结构或算法是什么意思 | [术语表](./glossary.md)、[公式表](./formulas.md) | [系统与执行架构](./architecture.md) |
| 当前支持哪些模型，其他 GGUF 为什么不能直接替换 | [模型基线与扩展顺序](./model_support.md) | [Chapter01 输入要求](../chapter01_tensor_loader/04_input_contract_and_tokenizer.md)、[多模态扩展教程](../extensions/multimodal/intro.md) |
| 请求怎样经过服务、调度和模型执行 | [系统与执行架构](./architecture.md#请求经过哪些阶段) | [Chapter06 Runtime](../chapter06_generation/next/intro.md) |
| 同一算子怎样切换 CPU 与 GPU | [CPU 主线与 GPU 扩展](./architecture.md#cpu-主线与-gpu-扩展) | [CUDA 安装与配置](../guides/setup_cuda.md) |
| 长上下文结果异常或 K/V 没有释放 | [Chapter06 K/V 语义](../chapter06_generation/next/06_single_request_kv_semantics.md) | [Chapter07 分页缓存](../chapter07_kv_cache/next/intro.md) |
| 多请求停滞、调度不公平或终态资源残留 | [Chapter08 调度](../chapter08_scheduling/next/intro.md) | [Chapter10 服务](../chapter10_serving/next/intro.md) |
| 性能下降，需要定位计算、访存或同步热点 | [Chapter09 性能剖析](../chapter09_performance/next/05_profiling_tools.md) | [实验资产](./experiment_assets.md) |
| HTTP、SSE、取消或超时行为异常 | [Chapter10 Transport](../chapter10_serving/next/04_transport.md) | <a href="https://github.com/datawhalechina/cpp-llm-engine/blob/main/code/chapter10_serving/README.md"><code>code/chapter10_serving/README.md</code></a> |
| 一章有哪些代码、验证和实验入口 | [使用说明](https://github.com/datawhalechina/cpp-llm-engine/blob/main/USAGE.md) | [实验资产](./experiment_assets.md)、[Chapter09 正确性与负载](../chapter09_performance/next/01_correctness_workload.md)、<a href="https://github.com/datawhalechina/cpp-llm-engine/blob/main/code/testing/README.md">公共测试基础设施</a> |
| 教学实现怎样对应成熟框架 | [系统与执行架构](./architecture.md) | [生态项目卡](./ecosystem/README.md) |
| 想先用短代码理解执行、分页缓存和 Radix 调度 | [Tiny-VLLM](./ecosystem/teaching_projects.md#tiny-vllm)、[nano-vLLM](./ecosystem/teaching_projects.md#nano-vllm)、[Mini-SGLang](./ecosystem/teaching_projects.md#mini-sglang) | 再进入 [生产级推理框架](./ecosystem/production_frameworks.md) |
| 主线完成后继续学习什么 | [生态项目索引](./ecosystem/README.md#完成主线后怎样继续) | 从对应章节的框架对照入口进入源码 |
