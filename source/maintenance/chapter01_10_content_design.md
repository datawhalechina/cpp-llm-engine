# Chapter01–10 主图与小节设计矩阵

本矩阵用于统一章节边界、intro 主图和正文编号。Chapter01–05 已进入正式正文；Chapter06–10 的目标正文暂存于 `next/`，完成导航切换前不删除旧页面。

## 主图设计

每章 intro 只保留一张主图，同时回答“本章在推理系统中的位置”和“本章输入、处理、输出是什么”。算法步骤、内存布局、状态机和性能时间线进入对应正文细节图。

| 章节 | 主题 | Intro 主图 | 主图必须表达 |
| --- | --- | --- | --- |
| Chapter01 | 输入、权重与模型配置 | `chapter01-main.svg` | 文件/metadata/tensor/tokenizer/config 到可执行模型对象 |
| Chapter02 | Tensor、Storage 与 View | `chapter02-main.svg` | Storage 所有权、shape/stride/offset 和 View/Materialize |
| Chapter03 | Operator、MatMul 与 Backend | `chapter03-main.svg` | Operator contract 到 Backend/Kernel，再到 correctness 验证 |
| Chapter04 | Attention、FFN 与 Decoder | `chapter04-main.svg` | Chapter01–03 产物怎样组装成 Decoder Layer 和真实模型层 |
| Chapter05 | Quantization | `chapter05-main.svg` | 量化权重从加载、反量化/直接计算到误差和后端选择 |
| Chapter06 | Generation Core | `chapter06-generation-core.svg` | 单请求 Prefill/Decode/采样/停止/流式输出，不包含多请求选择 |
| Chapter07 | KV Cache and Memory Management | `chapter07-kv-cache.svg` | 逻辑词元到 BlockTable、物理槽位、payload 和 Attention 读取 |
| Chapter08 | Scheduling and Batching | `chapter08-scheduling.svg` | running 请求与预算到 StepBatchPlan、缓存事务和执行提交 |
| Chapter09 | Performance Evaluation and Optimization | `chapter09-performance.svg` | correctness gate、固定 workload、测量、归因、A/B 与决策闭环 |
| Chapter10 | Serving | `chapter10-serving.svg` | Transport、准入、waiting/running、Scheduler、写回和统一释放 |

## 正文小节

| 章节 | 小节设计 | 状态 |
| --- | --- | --- |
| Chapter01 | 01.1 Tensor 对象；01.2 教学权重格式；01.3 模型加载与验证；01.4 输入契约与 Tokenizer；01.5 模型 metadata 与 GGUF；01.6 加载器框架对照 | 正式正文 |
| Chapter02 | 02.1 共享 Storage；02.2 reshape/transpose；02.3 slice；02.4 单元测试；02.5 Tensor/View 框架对照 | 正式正文；N 维与非连续 Tensor 已接入现有页面 |
| Chapter03 | 03.1 Operator 基类；03.2 MatMul 设计；03.3 CPU reference；03.4 cuBLAS backend；03.5 Python/PyTorch 独立复算；03.6 复杂度；03.7 Runtime backend 交接 | 正式正文 |
| Chapter04 | 04.1 Multi-head Attention；04.2 Norm/Activation；04.3 FFN；04.4 Decoder Layer；04.5 首层权重；04.6 真实 TinyLlama Decoder；04.7 框架对照 | 正式正文 |
| Chapter05 | 05.1 量化基础；05.2 Tensor 量化表达；05.3 量化权重加载；05.4 Dequantize；05.5 Quantized Linear；05.6 精度验证；05.7 框架对照；05.8 生成 Runtime 中的位置 | 正式正文 |
| Chapter06 | 06.1 请求/Executor 契约；06.2 Prefill/Decode；06.3 单请求循环；06.4 生命周期；06.5 采样与停止；06.6 history/KV 语义；06.7 streaming/error；06.8 真实 TinyLlama；06.9 框架对照 | `next/` 目标正文 |
| Chapter07 | 07.1 K/V shape；07.2 连续基线；07.3 Block/Page；07.4 BlockTable/Pool；07.5 Paged payload；07.6 事务；07.7 Paged Attention；07.8 Prefix Cache；07.9 真实模型迁移；07.10 框架对照 | `next/` 目标正文 |
| Chapter08 | 08.1 调度契约与 Round-robin；08.2 StepBatchPlan 与 Continuous Batching；08.3 Chunked Prefill、优先级与公平性；08.4 真实 Tensor batch 与缓存事务；08.5 Overlap；08.6 框架对照 | `next/` 正文 |
| Chapter09 | 09.1 Correctness/workload；09.2 Benchmark 入口；09.3 指标；09.4 统计/schema；09.5 Profiling 工具；09.6 热点归因；09.7 单变量 A/B；09.8 决策/回归；09.9 框架对照 | `next/` 目标正文 |
| Chapter10 | 10.1 ServiceCore；10.2 admission/容量/排队；10.3 终态释放；10.4 Transport；10.5 同步 Completion；10.6 SSE/断连/背压；10.7 并发/Worker；10.8 可观测性；10.9 真实模型验证；10.10 框架对照 | `next/` 目标正文 |

## 章间依赖

`Chapter01 -> Chapter02 -> Chapter03 -> Chapter04 -> Chapter05 -> Chapter06 -> Chapter07 -> Chapter08 -> Chapter09 -> Chapter10` 是完整学习顺序。实现依赖不要求 Chapter09 位于 Chapter10 调用链中：Performance 观察 Chapter06–10 的固定 workload，Serving 的执行依赖固定为 `Chapter10 -> Chapter08 -> Chapter07/Chapter06`。

| 交接 | 上一章交付 | 下一章使用 |
| --- | --- | --- |
| 01 → 02 | Tensor 数据与模型 metadata | Storage/View 语义 |
| 02 → 03 | shape/stride/offset 与连续化 | Operator 输入契约 |
| 03 → 04 | Backend/Kernel 和 MatMul | Attention/FFN/Decoder |
| 04 → 05 | 浮点 Decoder 路径 | 量化权重替换与核对 |
| 05 → 06 | 可执行模型后端 | 逐词元 Runtime |
| 06 → 07 | 单请求 K/V 增长语义 | 分页存储与资源事务 |
| 07 → 08 | reserve/commit/rollback/release | Step 选择与执行编排 |
| 08 → 09 | 固定调度策略和 workload | Benchmark/Profiling 证据 |
| 08 → 10 | Scheduler 公共接口 | 在线请求执行 |
| 09 → 10 | 服务 workload 和指标方法 | 在线性能验证 |
