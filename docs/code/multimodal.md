# 多模态 CPU 输入扩展代码导航

这组代码把图片解码、IDEFICS3 动态切图、SigLIP Vision Encoder、Projector 和 SmolVLM 文本 embedding 组合成统一序列，并通过 Engine adapter 接入 Chapter06 Runtime 与 Chapter10 Service Core。默认测试使用小型 fixture；真实模型验证需要 SmolVLM GGUF、mmproj 和测试图片。

## 阅读与验证入口

学习时先沿输入链路阅读，再进入真实权重和服务闭环。完整文件清单、CMake 选项与真实模型命令由代码目录 README 维护。

| 顺序 | 资产 | 看到的产物 |
| --- | --- | --- |
| 1 | <a href="/_code/multimodal/src/image_preprocessor.cpp"><code>image_preprocessor.cpp</code></a> | resize、动态切图和 patch tensor |
| 2 | <a href="/_code/multimodal/src/vision_encoder.cpp"><code>vision_encoder.cpp</code></a> | SigLIP block 与完整 encoder 输出 |
| 3 | <a href="/_code/multimodal/src/vision_projector.cpp"><code>vision_projector.cpp</code></a> | patch merge 与 LLM width 投影 |
| 4 | <a href="/_code/multimodal/src/input_embedding.cpp"><code>input_embedding.cpp</code></a> | 文本前缀、视觉 embedding、文本后缀拼接 |
| 5 | <a href="/_code/multimodal/examples/vision_projector_inspect.cpp"><code>vision_projector_inspect.cpp</code></a> | 真实 SmolVLM Decoder 与 ServiceCore 闭环 |
| 6 | <a href="/_code/multimodal/reference/benchmark_smolvlm_service.py"><code>benchmark_smolvlm_service.py</code></a> | 重复采集、阶段分布和正确性闸门 |
| 7 | <a href="/_code/multimodal/results/smolvlm_service_cpu_benchmark.json"><code>smolvlm_service_cpu_benchmark.json</code></a> | 两轮真实服务样本与可离线复查统计 |
| 8 | <a href="/_code/multimodal/README.md"><code>code/multimodal/README.md</code></a> | 构建选项、测试矩阵、模型路径和 golden 命令 |
