# Guides 开发参考

这里放的是写代码、接数据、做迁移时最常查的工程参考。它面向“正在实现功能”的读者，而不是纯概念阅读者。

## 内容范围

- [phase1_cpu_learning_plan.md](./phase1_cpu_learning_plan.md)：六个 CPU 主线任务、可选 GPU 初探和逐任务验收标准
- [learning_record_template.md](./learning_record_template.md)：保存环境、命令、正确性和解释的学习验收记录
- [cmake_guide.md](./cmake_guide.md)：CMake 常用技巧
- [setup_cuda.md](./setup_cuda.md)：CUDA 安装与配置
- [gguf_parser.md](./gguf_parser.md)：GGUF 解析与加载器实现
- [vllm_serving_baseline.md](./vllm_serving_baseline.md)：可选 vLLM Docker 服务基线，不属于 Chapter00 必读路径
- [multimodal_inference.md](./multimodal_inference.md)：SmolVLM 图片预处理、Vision Encoder、Projector 与现有 Runtime 的交接路线

## 怎么用

- 开发时遇到构建、环境或模型格式问题，先看这里
- 这些页面是工程参考，不是主线正文
- 如果你还在选学习环境或 GPU 路径，先看 [docs/guide.md](/guide) 里的硬件与环境说明
- 如果你在读主线教程，只在遇到具体工程问题时回来看
