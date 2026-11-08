# Reference 查阅索引

Reference 是按问题查阅的资料区，不是 Chapter00–10 之后必须顺序阅读的新章节。先确定要查的是概念、模型、系统结构、验证证据还是成熟框架，再进入对应页面。

| 要解决的问题 | 首选入口 | 继续查阅 |
| --- | --- | --- |
| 一个术语、公式或算法是什么意思 | [术语表](./glossary.md)、[公式表](./formulas.md) | [数据结构与算法资产表](./data_structures_algorithms.md)、[常见问题](./faq.md) |
| 当前支持哪些模型，其他 GGUF 为什么不能直接替换 | [模型基线与扩展顺序](./model_support.md) | [Chapter01 输入契约](../chapter01_tensor_loader/04_input_contract_and_tokenizer.md)、[多模态学习指南](../guides/multimodal_inference.md) |
| 引擎对象怎样分层，CPU 与 GPU 怎样协作 | [系统与执行分层](./inference_engine_systems.md) | [CPU/GPU 架构](./cpu_gpu_architecture.md)、[内存管理](./memory_management.md)、[Chapter06 Runtime](../chapter06_generation/next/intro.md) |
| 一章有哪些文档、代码和验证入口 | [章节资产表](./chapter_assets.md) | [实验资产](./experiment_assets.md)、[Chapter09 正确性与工作负载](../chapter09_performance/next/01_correctness_workload.md)、<a href="../_code/testing/README.md">公共测试基础设施</a> |
| 教学实现怎样对应成熟框架 | [框架对照总表](./framework_comparison.md) | [生态项目卡](./ecosystem/README.md) |
| 主线完成后继续学习什么 | [生态项目索引](./ecosystem/README.md#完成主线后怎样继续) | 从对应章节的框架对照入口进入源码 |
