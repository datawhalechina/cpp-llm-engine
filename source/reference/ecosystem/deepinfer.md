# DeepInfer 大规模生产系统参照

## 为什么读

[DeepInfer](https://github.com/deepseek-ai/DeepInfer) 展示 DeepSeek 模型在大规模服务中如何联合考虑模型结构、kernel、并行、异步调度和部署拓扑。它适合在掌握单机 Runtime 后建立集群级工程坐标。

## 重点看什么

- Python Engine 与 C++ Scheduler、BlockManager 等核心对象的分工
- 模型、采样、并行和异步调度怎样进入统一执行引擎
- Vision Encoder、Prefill、Decode 分离后，资源和指标怎样按阶段管理
- 开源离线引擎与完整在线服务基础设施之间仍有哪些缺口

当前项目面向特定 DeepSeek 模型和大规模加速器环境，不能作为单消费级 GPU 教程的运行 baseline，也不能用其集群 benchmark 与本地结果直接比较。

## 对应教程章节

- Chapter08：异步调度与批量执行边界
- Chapter09：分阶段 workload、指标和联合优化
- Chapter10：Engine、Service 与集群基础设施边界
- 多模态进阶：本地 Vision 路径与生产阶段分离
