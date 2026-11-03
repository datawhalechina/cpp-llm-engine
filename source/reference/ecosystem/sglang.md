# SGLang 前缀缓存与生产调度

## 为什么读

[SGLang](https://github.com/sgl-project/sglang) 同时覆盖推理 Runtime、结构化生成和服务系统。对本教程最有价值的是前缀 KV 复用怎样影响请求排序、缓存生命周期和资源预算，而不只是上层应用 workflow。

## 重点看什么

- RadixAttention/Radix Cache 怎样表示和复用可变长度共享前缀
- LPM 怎样把前缀命中信息用于等待队列排序
- Chunked Prefill 怎样和 token、KV、延迟预算共同工作
- HiCache 怎样把前缀 KV 扩展到 GPU、主机和外部存储层级

LPM 是调度策略，Radix Cache 是缓存与索引机制，HiCache 是多级存储扩展；三者相互协作，但不能写成同一个算法。本教程先保证请求级 K/V、block ownership 和释放语义正确，再讨论跨请求复用。

## 源码坐标

| 对象 | 官方入口 | 阅读问题 |
| --- | --- | --- |
| Scheduler | [`scheduler.py`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/scheduler.py) | token/cache budget 怎样进入调度循环 |
| Radix Cache | [`radix_cache.py`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/mem_cache/radix_cache.py) | prefix match、锁定、插入和淘汰怎样改变 ownership |
| SRT Runtime | [`python/sglang/srt`](https://github.com/sgl-project/sglang/tree/main/python/sglang/srt) | API、scheduler、worker 和 cache 怎样分层 |
| Benchmark | [`benchmark`](https://github.com/sgl-project/sglang/tree/main/benchmark) | serving、cache 和 workload 条件怎样定义 |

## 对应教程章节

- Chapter06：Prefill/Decode、Chunked Prefill、Scheduler 和 Block KV
- Chapter07-08：前缀命中 workload、TTFT 和资源成本
- Chapter09：admission、前缀复用、缓存回收和服务指标
