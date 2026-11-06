# Chapter00–10 教学设计与职责边界审核

本文审核目标章节，不替代学习者导读。审核对象包括正式发布的 Chapter00–05，以及仍位于 `next/` 的 Chapter06–10 重构正文。

## 总体结论

推荐主线已经形成连续依赖：

```text
Chapter00 真实外部基线
  -> Chapter01 输入、配置与权重
  -> Chapter02 Tensor Storage/View
  -> Chapter03 Operator/Backend
  -> Chapter04 Decoder 数学与结构
  -> Chapter05 量化权重执行
  -> Chapter06 单请求 Generation Core
  -> Chapter07 KV Cache 与物理内存
  -> Chapter08 Scheduling 与 Batching
  -> Chapter09 性能证据与优化决策
  -> Chapter10 Serving
```

这条顺序符合“先得到真实结果，再拆开数据、计算、模型、运行时和服务”的学习过程。当前主要风险不是章节顺序，而是正式站点仍发布旧 Chapter06–09，新 Chapter06–10 尚未切换；学习者看到的章节职责与目标设计不一致。

## 章节职责

| 章节 | 只负责什么 | 交付给下一章什么 | 不应继续承担什么 |
| --- | --- | --- | --- |
| Chapter00 | 用 llama.cpp 和 TinyLlama 建立真实推理基线 | 可重复的模型、prompt、token 与输出参照 | 自研 Loader、算子或多框架安装教程 |
| Chapter01 | 把文本和模型文件转换为 Token、配置、权重与模型布局 | Tensor、ModelConfig、Tokenizer 契约和权重索引 | Tensor View 运算、Decoder 计算和请求调度 |
| Chapter02 | 解释 Storage、shape、stride、offset、View 与连续化 | 可交给算子的公共只读/可写 View | MatMul 语义、设备后端和自动批量计算 |
| Chapter03 | 建立 Operator、Backend、Kernel 分层并完成 MatMul | 稳定计算请求、CPU reference 和可选 CUDA 对照 | Decoder 拓扑、量化格式和通用 GPU 优化大全 |
| Chapter04 | 解释 Attention、FFN、归一化、残差和 Decoder 结构 | 单层与多层 Decoder 的数学顺序和执行契约 | 请求生命周期、物理分页缓存和调度策略 |
| Chapter05 | 让 packed 低比特权重进入 Linear，并验证误差 | 可供真实 Decoder 使用的量化 Backend | Generation、KV 调度或服务容量策略 |
| Chapter06 | 固定单请求 Prefill、Decode、采样、停止和回调语义 | Request、Runtime、Executor 和终态契约 | 多请求选择、物理 K/V 分配和 HTTP 生命周期 |
| Chapter07 | 管理 K/V 的连续基线、分页寻址、数据面和事务 | BlockTable、KvBlockPool、分页读写与回滚接口 | 选择哪个请求执行、准入排队和网络响应 |
| Chapter08 | 根据请求状态和预算生成 Step plan，并组织执行批次 | 可复现的候选顺序、批次输入和提交结果 | 服务 waiting queue、Transport 和性能结论 |
| Chapter09 | 固定 workload、指标、统计、Profiling 和 A/B 决策 | 可比较的证据记录和 KEEP/REJECT 结论 | 请求生命周期所有权或网络实现 |
| Chapter10 | 组合准入、队列、Worker、HTTP/SSE、终态释放和指标 | 可运行的单机推理服务闭环 | 重写 Scheduler、KV 数据面或模型数学 |

## 关键边界问题

### Chapter04 与 Chapter05

Chapter04 的教学 Decoder 可以独立完成，但真实 TinyLlama Q4_K/Q6_K 执行依赖 Chapter05。当前 Chapter04 把真实 22 层 Decoder 列为本章 L2 验收，同时要求先完成下一章，破坏了逐章闭环。

调整方式：Chapter04 只验收浮点教学 Decoder、真实结构映射和执行器骨架；完整 TinyLlama token golden 移到 Chapter05 的跨章验收。Chapter04 可以保留真实模型页面，但页面产物应改为“结构和权重映射可检查”，不再要求完成量化执行。

### Chapter04、Chapter06 与 Chapter07

三章都出现 K/V，但观察层级不同：Chapter04 说明 Attention 为什么需要历史 K/V；Chapter06 把模型 Context 视为请求专属执行状态；Chapter07 才拥有 K/V 的物理布局、分页、事务和释放。正文应统一使用这组三层定义，避免把 Chapter06 的 token history 描述成模型 K/V。

### Chapter08 与 Chapter10

Chapter08 决定“本 step 执行谁、消耗多少预算”；Chapter10 决定“请求是否进入系统、处于 waiting 还是 running、怎样取消和写回”。Serving 可以调用 Scheduler，但不能保存第二份调度状态或自行分配 K/V。

### Chapter09 的位置

性能方法放在 Serving 前是合理的：它先用 Engine、KV 和 Scheduler workload 建立证据方法，Chapter10 再复用同一 schema 测服务负载。TTFT、ITL 等指标必须说明测量起点；Runtime 指标与包含网络排队的服务指标不能直接混用。

## 文档成熟度

| 章节 | 正文完整性 | 文档与代码衔接 | 发布状态 | 结论 |
| --- | --- | --- | --- | --- |
| Chapter00 | 高 | 高 | 正式 | 入口清楚；继续保持只提供 llama.cpp 基线 |
| Chapter01 | 高 | 高 | 正式 | Loader、Tokenizer、GGUF 与真实模型衔接完整 |
| Chapter02 | 高 | 高 | 正式 | View/Storage 边界清楚，跨章测试充分 |
| Chapter03 | 高 | 高 | 正式 | CPU reference、Backend 和 CUDA 对照完整 |
| Chapter04 | 高 | 高 | 正式 | 内容扎实；真实模型验收需解除对 Chapter05 的反向依赖 |
| Chapter05 | 高 | 高 | 正式 | 量化格式、执行和误差证据完整，可接收 Chapter04 真实验收 |
| Chapter06 | 高 | 高 | `next/` | 新职责清楚；旧正文仍混有 Scheduler、KV 和 Serving 内容 |
| Chapter07 | 中高 | 中 | `next/` | 分页教学闭环成立；真实 TinyLlama 分页 Context 尚未接通 |
| Chapter08 | 中高 | 中 | `next/` | 调度概念完整；真实分页 batch 与 overlap 仍是规划能力 |
| Chapter09 | 中 | 中低 | `next/` | 方法框架成立，但采集脚本、结果资产和正文代码跳转仍指向旧章节 |
| Chapter10 | 中高 | 高 | `next/` | ServiceCore、HTTP、Worker 和真实模型已验证；正文缺少直接代码入口，异步背压尚未实现 |

成熟度不能只按篇幅判断。Chapter07、08 的正文已经足够解释教学实现，未实现的生产扩展也有明确标记；真正阻塞正式切换的是 Chapter09 资产迁移和 Chapter06–10 的统一导航、代码链接与验收入口。

## 调整优先级

1. 修正 Chapter04/05 验收边界，把真实量化 TinyLlama 的 token golden 收口到 Chapter05。
2. 完成 Chapter09 旧 Benchmark/Profiling 脚本和结果 schema 迁移，正文直接链接新代码资产。
3. 为 Chapter08–10 正文补充可点击的代码与测试入口，不再只写文件名。
4. 对 Chapter06–10 执行一次“正文声明—代码对象—测试证据”逐页核对。
5. 一次性切换 VitePress、README、USAGE 和 guide 到 Chapter06–10 新导航。
6. 导航切换验证通过后，再删除旧 Chapter06 调度正文、Chapter07 Benchmark、Chapter08 Profiling 和 Chapter09 Serving 页面。
7. 将真实分页 Context、真实 Tensor batch、Overlap 和异步背压作为切换后的增强任务，不用它们阻塞教学主线发布。

## 发布门槛

Chapter06–10 可以进入正式导航的最低条件：

- 每章 intro 只有一张承担定位和内容概要的主图。
- 每个正文页面能从概念进入对应代码、测试或明确的规划状态。
- 每章默认 CPU 入口可离线运行，真实模型和 CUDA 入口明确标为可选层级。
- 相邻章节只通过公开契约交接，不依赖旧章节同名实现。
- 文档构建、全局回归和 Release contracts 全部通过。

