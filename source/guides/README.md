# 学习与复现指南

这里保存跨章节的学习计划、环境操作和复现流程。章节原理留在正文，代码结构和命令细节留在对应 `code/<chapter>/README.md`。

## 内容范围

- [phase1_cpu_learning_plan.md](./phase1_cpu_learning_plan.md)：六个 CPU 主线任务、可选 GPU 初探和学习记录模板
- [setup_cuda.md](./setup_cuda.md)：CUDA 安装与配置
- [vllm_serving_baseline.md](./vllm_serving_baseline.md)：可选 vLLM Docker 服务基线，不属于 Chapter00 必读路径
- [multimodal_inference.md](./multimodal_inference.md)：SmolVLM 图片预处理、Vision Encoder、Projector 与现有 Runtime 的交接路线

## 怎么用

- 需要按产物推进 CPU 主线时使用学习计划。
- 需要 CUDA、多模态或 vLLM 环境时进入对应专题。
- CMake 前置能力见 [Foundation](../foundation/intro.md)，GGUF 解析见 [Chapter01](../chapter01_tensor_loader/05_model_metadata_and_gguf.md)。
