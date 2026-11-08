# Reference 查阅索引

这里汇总概念解释、系统对比、术语公式、实验资产和外部项目阅读入口。

## 内容范围

- [framework_comparison.md](./framework_comparison.md)：训练框架、推理引擎、推理服务和本教程的系统位置
- [inference_engine_systems.md](./inference_engine_systems.md)：推理引擎运行时、内存、profiling、输入链路和服务化补强图
- [execution_layers.md](./execution_layers.md)：算子、Backend、Executor 与 Runtime 的分层
- [chapter_assets.md](./chapter_assets.md)：各章正文、代码、图、公式和验证入口
- [experiment_assets.md](./experiment_assets.md)：性能实验、正确性 golden、结果文件和复现入口
- [data_structures_algorithms.md](./data_structures_algorithms.md)：跨章节数据结构、算法、代码入口和验证产物总表
- [cpu_gpu_architecture.md](./cpu_gpu_architecture.md)：CPU 主线、公共契约、GPU 扩展和证据闭环的统一架构
- [memory_management.md](./memory_management.md)：Tensor、workspace、KV Cache 和 staging 的资源生命周期
- [runtime_systems.md](./runtime_systems.md)：推理引擎运行时边界
- [profiling.md](./profiling.md)：benchmark 之后的瓶颈解释
- [input_pipeline.md](./input_pipeline.md)：推理引擎输入链路
- [multimodal_inference.md](./multimodal_inference.md)：图片动态切图、视觉编码和 LLM 接入链路
- [model_support.md](./model_support.md)：默认模型、稠密模型兼容目标与 MoE 建设顺序
- [service_basics.md](./service_basics.md)：推理引擎服务化基础
- [regression_testing.md](./regression_testing.md)：回归测试和版本化基线
- [formulas.md](./formulas.md)：Tensor、模型、Runtime、性能和多模态公式索引
- [glossary.md](./glossary.md)：核心术语表
- [faq.md](./faq.md)：常见问题
- [further_reading.md](./further_reading.md)：主线之外继续看什么
- [ecosystem/README.md](./ecosystem/README.md)：按工程问题选择外部项目卡片

## 怎么用

- 查术语和公式时，进入 `glossary.md` 或 `formulas.md`
- 查章节代码、图片和验证入口时，进入 `chapter_assets.md`
- 查性能实验和复现命令时，进入 `experiment_assets.md`
- 查模型支持状态和扩展顺序时，进入 `model_support.md`
- 对照成熟框架时，先读 `framework_comparison.md`，再进入 `ecosystem/`
