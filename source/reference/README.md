# 参考资料索引

Reference 是按问题查阅的资料区，不是 Chapter00–10 之后必须顺序阅读的新章节。先确定要查的是概念、模型、系统结构、验证证据还是成熟框架，再进入对应页面。

| 要解决的问题 | 首选入口 | 继续查阅 |
| --- | --- | --- |
| 一个术语、公式、数据结构或算法是什么意思 | [术语表](./glossary.md)、[公式表](./formulas.md) | [教程资产索引](./tutorial_assets.md) |
| 当前支持哪些模型，其他 GGUF 为什么不能直接替换 | [模型基线与扩展顺序](./model_support.md) | [Chapter01 输入要求](../chapter01_tensor_loader/04_input_contract_and_tokenizer.md)、[多模态学习计划](../guides/multimodal_learning_plan.md) |
| 请求怎样经过服务、调度和模型执行 | [系统与执行架构](./architecture.md#请求经过哪些阶段) | [Chapter06 Runtime](../chapter06_generation/next/intro.md) |
| 同一算子怎样切换 CPU 与 GPU | [CPU 主线与 GPU 扩展](./architecture.md#cpu-主线与-gpu-扩展) | [CUDA 安装与配置](../guides/setup_cuda.md) |
| 一章有哪些代码、验证和实验入口 | [教程资产索引](./tutorial_assets.md) | [实验资产](./experiment_assets.md)、[Chapter09 正确性与工作负载](../chapter09_performance/next/01_correctness_workload.md)、<a href="https://github.com/datawhalechina/cpp-llm-engine/blob/main/code/testing/README.md">公共测试基础设施</a> |
| 教学实现怎样对应成熟框架 | [系统与执行架构](./architecture.md) | [生态项目卡](./ecosystem/README.md) |
| 主线完成后继续学习什么 | [生态项目索引](./ecosystem/README.md#完成主线后怎样继续) | 从对应章节的框架对照入口进入源码 |
