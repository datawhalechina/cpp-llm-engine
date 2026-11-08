# 09.9 把教学证据链映射到成熟框架

成熟框架的性能工具都需要明确工作负载、指标、阶段范围、原始结果和决策。本章先用统一证据结构完成可重复实验，再通过 `reference/ecosystem` 查找容易随版本变化的命令和源码入口。

## 对象映射

| 对照对象 | 重点观察 | 与本章的连接 |
| --- | --- | --- |
| llama.cpp | `llama-bench`、线程与后端参数、模型量化结果 | 单机 CPU/GPU 固定配置与重复测量 |
| vLLM | 离线吞吐、在线服务指标、调度与 Kernel Profiling | workload、TTFT/ITL、批处理和服务负载 |
| SGLang | 前缀缓存、调度策略和服务 Benchmark | 缓存命中、调度变化与端到端结果 |
| TensorRT-LLM | 构建配置、Inflight Batching、CUDA/Nsight 证据 | GPU 构建身份、Kernel 和端到端分层 |
| DeepInfer | 推理阶段和硬件执行证据 | 将新后端放入相同正确性与 A/B 契约 |

## 教学实现与生产实现

| 教学实现先保证 | 成熟工具通常还需要 |
| --- | --- |
| 单机、固定模型和少量 workload | 数据集采样、并发到达分布和长时间压测 |
| 基础延迟、吞吐和内存 | 百分位、SLO、功耗与多租户隔离 |
| CPU/CUDA 阶段归因 | 多进程、多 GPU、通信和跨节点时间线 |
| 单变量 A/B 与人工登记 | 自动基线选择、趋势存储和回归告警 |

## 继续阅读源码

阅读成熟工具时先回答三个问题：工作负载在哪里固定，原始样本在聚合前保存在哪里，正确性闸门与性能决策是否分开。

| 项目卡 | 本章使用它回答什么 |
| --- | --- |
| [llama.cpp](../../reference/ecosystem/local_and_delivery.md#llama-cpp) | 单机 CPU/GPU 如何固定模型、线程和量化条件 |
| [vLLM](../../reference/ecosystem/production_frameworks.md#vllm) | 离线吞吐与在线 TTFT/ITL 如何区分 workload |
| [SGLang](../../reference/ecosystem/production_frameworks.md#sglang) | 前缀缓存和调度变化如何进入服务 Benchmark |
| [TensorRT-LLM](../../reference/ecosystem/production_frameworks.md#tensorrt-llm) | GPU 构建身份、Nsight 证据与端到端结果如何分层 |
