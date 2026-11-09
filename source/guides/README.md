# 学习与复现指南

这里保存跨章节的学习计划、环境操作和复现流程。CPU 是完整主线，GPU 和多模态在主线契约上扩展执行后端与输入路径。章节原理留在正文，代码结构和命令细节留在对应 `code/<chapter>/README.md`。

## 内容范围

- [cpu_learning_plan.md](./cpu_learning_plan.md)：六个 CPU 主线任务和学习验收记录模板
- [gpu_learning_plan.md](./gpu_learning_plan.md)：六个 GPU 扩展任务，从设备资源推进到真实模型服务
- [multimodal_learning_plan.md](./multimodal_learning_plan.md)：六个多模态扩展任务，从图片预处理推进到统一 Runtime
- [setup_cuda.md](./setup_cuda.md)：CUDA 安装与配置
- [vllm_service_quickstart.md](./vllm_service_quickstart.md)：用独立 Docker 容器体验 vLLM 的同步与流式服务

## 怎么用

- 第一次学习按 CPU 计划建立完整推理服务，再按兴趣选择 GPU 或多模态计划。
- 安装或排查 CUDA 时使用环境指南；体验外部成熟服务时使用 vLLM 快速体验。
- CMake 前置能力见 [Foundation](../foundation/intro.md)，GGUF 解析见 [Chapter01](../chapter01_tensor_loader/05_model_metadata_and_gguf.md)。
