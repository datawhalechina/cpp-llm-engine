# Mini-SGLang 调度与缓存骨架

## 为什么读

[Mini-SGLang](https://github.com/sgl-project/mini-sglang) 将现代 LLM serving 的调度和缓存机制保留在相对紧凑的 Python 代码中，适合观察 Prefill、Decode、Radix Cache 和 overlap scheduling 如何协作。

## 重点看什么

- Prefill 与 Decode 队列怎样竞争 token 和 cache 预算
- Radix Cache 怎样匹配共享前缀并管理复用生命周期
- overlap scheduling 怎样重叠 CPU 调度和 GPU 执行

它依赖 CUDA kernel 和外部高性能组件；教程先用 CPU 状态机和确定性测试固定语义，再阅读其异步执行路径。

## 源码坐标

| 对象 | 官方入口 | 阅读问题 |
| --- | --- | --- |
| Scheduler | [`scheduler.py`](https://github.com/sgl-project/mini-sglang/blob/main/python/minisgl/scheduler/scheduler.py) | normal/overlap loop 怎样交换 batch 和执行结果 |
| 项目全貌 | [Mini-SGLang](https://github.com/sgl-project/mini-sglang) | 调度、缓存和模型执行怎样组成最小系统 |

## 对应教程章节

- Chapter06：Prefill/Decode、Chunked Prefill 和 Scheduler
- Chapter09：请求 admission、缓存复用和资源释放
- Chapter07-08：调度 A/B workload 与重叠效果
