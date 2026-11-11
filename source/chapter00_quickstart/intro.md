# 00.0 建立真实推理基线

## 本章导读

Foundation 准备了构建和测试能力；本章先使用 `llama.cpp` 加载 TinyLlama GGUF，得到一次真实分词、模型执行和生成结果。掌握默认入口后，可以用 Qwen3 对照另一种稠密架构，用 Qwen1.5-MoE 观察专家模型，或用 DeepSeek-V2-Lite 观察 MLA 与 MoE 的组合。学习者完成 TinyLlama 后即可进入 Chapter01，其余模型按学习主题准备。

| 教学范围 | 成熟实现提供 | 深入入口 |
| --- | --- | --- |
| 构建本地基线、准备模型、运行一次 CPU 推理 | GGUF 加载器、分词器、模型执行和采样循环 | [`llama.cpp` 项目卡](../reference/ecosystem/local_and_delivery.md#llama-cpp) |

| 学习线 | 本章怎样学习 | 可验证产物 |
| --- | --- | --- |
| A 准备检查线 | 编译仓库内最小程序，确认 C++、CMake、路径和脚本入口可用 | `smoke check passed` 与明确的失败位置 |
| B 真实基线线 | 构建 llama.cpp，下载 TinyLlama GGUF，并运行一次真实分词与生成 | 可重复执行的真实 prompt、token 和生成结果 |
| C 架构扩展线 | 保持 llama.cpp 入口不变，切换 Qwen3、Qwen1.5-MoE 或 DeepSeek-V2-Lite GGUF | 稠密、MoE 与 MLA 模型的外部运行坐标 |

![Chapter 00 主图：真实推理基线](../assets/diagrams/chapter00-main.svg)

## 开始学习

先运行仓库内不依赖模型的快速检查：

```bash
bash code/chapter00_quickstart/run.sh
```

看到 `chapter00_quickstart` 输出 `smoke check passed` 后，说明连通性检查已经通过。按表格继续准备 `llama.cpp` 和 TinyLlama，第四步使用 <a href="https://github.com/datawhalechina/cpp-llm-engine/blob/main/code/chapter00_quickstart/run_real_inference.sh"><code>run_real_inference.sh</code></a> 执行真实模型。

| 顺序 | 路线 | 页面 | 读完后应看到什么 |
| --- | --- | --- | --- |
| 1 | A/B 公共 | [`00.1 检查本地环境`](./01_environment_setup.md) | C++、CMake 和 CPU 运行条件可用 |
| 2 | B | [`00.2 构建 llama.cpp 外部基线`](./02_clone_and_build_llamacpp.md) | `llama.cpp` 可执行文件构建完成 |
| 3 | B/C | [`00.3 选择并准备真实模型`](./03_download_tinyllama.md) | 默认 TinyLlama 可读取，并知道何时选择 Qwen3 或 MoE |
| 4 | B | [`00.4 运行第一次真实推理`](./04_first_inference.md) | 真实分词器处理输入文本，模型生成非空结果 |
| 5 | A/B 公共 | [`00.5 排查首次运行问题`](./05_troubleshooting.md) | 能按环境、路径、模型和运行参数排查失败 |

## 验证与下一步

真实模型基线建立后，进入 [`Chapter 01`](../chapter01_tensor_loader/intro.md)，开始实现教程自己的张量对象、模型加载器、元数据适配器和分词器接口。

| 验收层级 | 执行入口 | 通过条件 | 本章要求 |
| --- | --- | --- | --- |
| L1 默认 CPU | `bash code/chapter00_quickstart/run.sh` | 仓库快速检查编译成功并输出 `smoke check passed` | 必做 |
| L2 真实模型 | `bash code/chapter00_quickstart/run_real_inference.sh` | TinyLlama GGUF 被读取，真实分词器处理输入文本，模型生成结果并正常退出 | 必做 |
| L3 可选 CUDA | 使用 `llama.cpp/build-cuda` 运行同一真实推理脚本 | 外部 GPU 基线生成非空结果；不代表教程自研 CUDA backend | 选做 |
| L3 可选架构 | 运行 Qwen1.5-MoE、DeepSeek-V2-Lite 脚本或指定 Qwen3 GGUF | 外部框架识别目标架构并生成非空结果 | 学习模型适配时选做 |
