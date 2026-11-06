# nano-vLLM Python 算法骨架

## 为什么读

[nano-vLLM](https://github.com/GeeeekExplorer/nano-vllm) 用紧凑 Python 实现模型执行和 vLLM 风格 Runtime，适合先理解对象关系和控制流，再回到本教程处理 C++ ownership、错误契约和 backend。

## 重点看什么

- 模型层怎样组织权重和 forward
- Request、block、scheduler 和 model runner 怎样传递状态
- Prefix caching、CUDA Graph 等优化依赖哪些基础对象

本教程用短 Python 镜像解释算法，再用 C++ 实现 ownership、错误契约和 backend。

## 源码坐标

| 对象 | 官方入口 | 阅读问题 |
| --- | --- | --- |
| 项目代码 | [nano-vLLM](https://github.com/GeeeekExplorer/nano-vllm) | model、scheduler、block manager 和 runner 怎样传递状态 |

## 对应教程章节

- Chapter04：模型层与 Attention 结构
- Chapter06：Generation Runtime 与单请求循环
- Chapter07-08：K/V 资源管理、Scheduler 与批处理
- Chapter09：优化对象和 Profiling 观测点
