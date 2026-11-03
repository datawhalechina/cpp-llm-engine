# 07.7 对照成熟框架的基准方法

不同框架的默认工作负载、服务方式和指标口径并不相同。对照前先固定模型、输入输出长度、并发、硬件和正确性，再比较结果。

## 对象映射

| 教程证据 | 成熟基准中的对应项 | 先检查什么 |
| --- | --- | --- |
| 正确性门槛 | golden output、模型质量或请求成功率 | 性能是否建立在正确输出上 |
| prefill / decode 时间 | TTFT、TPOT、ITL | 指标起止点是否一致 |
| token/s | request/output throughput | 分母是模型时间、墙钟时间还是稳态窗口 |
| 双请求时间轴 | concurrency / serving workload | 请求到达方式和调度策略 |
| JSON 结果文件 | benchmark report | 模型、dtype、硬件和命令是否完整 |

## 教学实现与生产实现

| 本章可验证对象 | 成熟系统继续增加 |
| --- | --- |
| 固定 `1/2/4` 请求和平均耗时 | 开放环到达率、长时间稳态和 P50/P95/P99 |
| 教学 Round-robin 与真实双 worker | Continuous Batching、Tensor batch 和抢占 |
| 本机 CPU JSON | GPU 利用率、显存峰值、多 GPU 和跨节点结果 |
| 手工保存基线 | 结果数据库、趋势面板和持续性能回归 |

## 继续阅读源码

| 项目卡 | 在本章中的用途 |
| --- | --- |
| [llama.cpp](../reference/ecosystem/llama_cpp.md) | 本地 prompt processing 与 generation 基线 |
| [vLLM](../reference/ecosystem/vllm.md) | online、offline 和 latency workload |
| [SGLang](../reference/ecosystem/sglang.md) | serving 与 cache workload |
| [TensorRT-LLM](../reference/ecosystem/tensorrt_llm.md) | Engine 配置与 GPU benchmark |

阅读成熟框架结果时，先回答三件事：计时范围是否一致、请求到达方式是否一致、正确性和失败率是否一起报告。源码版本和易变入口由上述项目卡维护。
