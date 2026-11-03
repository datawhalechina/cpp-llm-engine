# Gemma 模型家族

## 解决什么问题

`Gemma` 适合作为一个具体的模型家族，用来做兼容性检查、演示和验证。

## 场景定位

- 适合看某个具体模型家族是如何落到 C++ 推理实现里的
- 适合做兼容性、量化和 benchmark 的验证对象
- 适合和 `llama.cpp` 形成“模型家族视角 + 通用运行时视角”的对照

## 工程策略

- 提供一个足够真实的模型家族作为测试对象
- 帮助验证架构和兼容性选择
- 在教程需要具体目标时，作为示例模型家族

## 官方入口

| 入口 | 用途 |
| --- | --- |
| [Gemma 模型文档](https://ai.google.dev/gemma/docs) | 查模型家族、结构和使用方式 |
| [gemma.cpp](https://github.com/google/gemma.cpp) | 观察具体模型家族到轻量 C++ Runtime 的映射 |

## 为什么和 `llama.cpp` 很适合一起看

`Gemma` 给教程提供了一个具体的模型家族锚点，`llama.cpp` 给教程提供了一个更宽的本地推理 baseline。两者一起看，读者就能在“具体模型实现细节”和“更通用的生产级运行时结构”之间来回切换。

如果目标是 quickstart 和 first-success 流程，`llama.cpp` 更适合先看；如果目标是模型家族验证和兼容性讨论，`Gemma` 更适合作为参考。

## 对应教程章节

`Gemma` 主要对应教程里的这些章节：

- `chapter01_tensor_loader`
- `chapter04_attention`
- `chapter05_quantization`
- `chapter07_benchmark`
