---
layout: home

hero:
  name: 从零构建 C++ LLM 推理引擎
  text: cpp-llm-engine
  tagline: 从 GGUF、Tensor、Decoder 到 KV Cache、调度与推理服务
  actions:
    - theme: brand
      text: 进入教程主入口
      link: /README
    - theme: alt
      text: 看阅读指南
      link: /guide
    - theme: alt
      text: English entry
      link: /en/README

features:
  - title: 中文主线
    details: 先把主线讲清楚，再用英文入口和术语表照顾非中文母语学习者。
  - title: 工程化入门
    details: 从 Foundation 到 Chapter10，依次连接权重、计算、Runtime、调度和服务。
  - title: 可验证主线
    details: 用单测、golden、benchmark 和 profiling 验证每个阶段的产物。
---

## 这是什么

这是一个面向中文学习者优先的 C++ LLM 推理教程，最终目标是构建可替换 Loader、Backend、KV Cache、Scheduler 和 Transport 的单机 DIY 推理服务框架。
如果你想先知道怎么读，先看 [阅读指南](/guide)；如果你想直接看主线，从 [教程主入口](/README) 进入。
环境和硬件建议集中在 [阅读指南](/guide)。

## 教程主线

这套教程按一条推理引擎主线往前推进，而不是按主题合集散着讲。

| 阶段 | 主要职责 |
| --- | --- |
| `chapter01` - `chapter03` | 输入、张量语义、算子边界 |
| `chapter04` - `chapter06` | Decoder、量化与生成运行时 |
| `chapter05` | 表示与成本 |
| `chapter07` | K/V Cache 与内存管理 |
| `chapter08` | 调度与批处理 |
| `chapter09` | 性能证据与优化闭环 |
| `chapter10` | ServiceCore、HTTP/SSE 与资源释放 |

完成主线后，按需查阅 `reference` 和 `guides`：

- `reference` 讲概念解释、术语和生态对照
- `guides` 讲构建、环境、迁移和工程方法

想先看训练框架、推理引擎和服务怎样分工，可以直接进[系统与执行架构](/reference/architecture)。

## 你会看到什么

- `foundation` 到 `chapter10_serving` 的主线
- 训练框架、推理框架和本教程的对照层
- 术语表、公式表和跨章节资产表
- 图示、表格、代码入口和验证结果

## 适合谁

- 想从零看懂推理引擎的人
- 想理解 runtime、KV Cache、benchmark 证据链的人
- 想把教程内容迁移到自己项目里的人
