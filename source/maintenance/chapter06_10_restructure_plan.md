# Chapter06–10 重构与迁移矩阵

本表供维护者拆分 Generation、KV Cache、Scheduling、Performance 和 Serving。迁移期间保留旧目录，先建立新文档与代码闭环，最后统一切换导航并删除旧资产。

## 目标结构

| 章节 | 学习问题 | 主要输出 | 不再承担 |
| --- | --- | --- | --- |
| Chapter06 Generation Core | 单个请求怎样从输入词元推进到停止 | 请求状态、Prefill/Decode、采样、流式词元 | 多请求选择、分块内存分配、网络服务 |
| Chapter07 KV Cache and Memory Management | K/V 怎样保存、寻址、提交和释放 | Cache Manager、BlockTable、KV payload、Paged Attention | 请求优先级和 Step batch 选择 |
| Chapter08 Scheduling and Batching | 每一步选择哪些请求和词元执行 | StepBatchPlan、预算、公平性、Tensor batch | K/V payload 所有权和 HTTP 生命周期 |
| Chapter09 Performance Evaluation and Optimization | 怎样证明结果正确并定位性能变化 | 固定工作负载、指标、Profiling、A/B 结论 | 在线请求接入和资源调度实现 |
| Chapter10 Serving | 怎样把 Runtime 暴露为可观测的持续服务 | ServiceCore、Transport、SSE、并发、指标 | StepBatchPlan、Paged KV 和算子实现 |

依赖方向固定为 `Generation Core -> KV Cache -> Scheduling -> Performance -> Serving`。Chapter08 可以调用 Chapter07 的 `reserve/commit/rollback/release`，Chapter10 可以配置并提交请求给 Chapter08；反向依赖不成立。

## 目标正文小节

| 章节 | 小节 | 主题与可验证对象 |
| --- | --- | --- |
| Chapter06 | 06.1 | 请求、生成配置和模型执行输入契约 |
| Chapter06 | 06.2 | Prefill 与 Decode 的输入、输出和状态差异 |
| Chapter06 | 06.3 | 单请求逐词元生成循环 |
| Chapter06 | 06.4 | 请求阶段、完成、取消和失败状态 |
| Chapter06 | 06.5 | 采样策略、停止词元和长度限制 |
| Chapter06 | 06.6 | 词元历史、位置和单请求模型 K/V 增长语义 |
| Chapter06 | 06.7 | 流式词元与错误传播 |
| Chapter06 | 06.8 | 真实 TinyLlama 单请求生成闭环 |
| Chapter06 | 06.9 | 成熟框架 Generation Core 对照 |
| Chapter07 | 07.1 | Attention 中 K/V 的 shape 与序列维度 |
| Chapter07 | 07.2 | 连续 KV Cache 基线 |
| Chapter07 | 07.3 | Page、Block、逻辑词元位置和物理槽位 |
| Chapter07 | 07.4 | BlockTable 与 KvBlockPool |
| Chapter07 | 07.5 | 分页 K/V payload 的读写与请求隔离 |
| Chapter07 | 07.6 | reserve、commit、rollback、trim 和 release 事务 |
| Chapter07 | 07.7 | Paged Attention 的页表寻址 |
| Chapter07 | 07.8 | Prefix Cache、Radix 索引与淘汰策略 |
| Chapter07 | 07.9 | 真实模型迁移与连续/分页结果核对 |
| Chapter07 | 07.10 | 成熟框架 Cache Engine 对照 |
| Chapter08 | 08.1 | Scheduler 契约与 Round-robin 基线 |
| Chapter08 | 08.2 | StepBatchPlan、Continuous Batching 与混合执行 |
| Chapter08 | 08.3 | Chunked Prefill、优先级与公平性 |
| Chapter08 | 08.4 | Tensor batch、真实模型与缓存事务 |
| Chapter08 | 08.5 | Overlap Scheduling 的状态版本 |
| Chapter08 | 08.6 | 成熟框架调度对照 |
| Chapter09 | 09.1 | Correctness Gate 与固定工作负载 |
| Chapter09 | 09.2 | Benchmark 入口、环境和预热 |
| Chapter09 | 09.3 | 延迟、吞吐、TTFT、ITL 和内存指标 |
| Chapter09 | 09.4 | 重复测量、统计聚合和结果 schema |
| Chapter09 | 09.5 | CPU/GPU Profiling 工具与使用顺序 |
| Chapter09 | 09.6 | 阶段耗时、带宽和 Kernel 热点 |
| Chapter09 | 09.7 | 单变量 A/B 实验 |
| Chapter09 | 09.8 | 保留、回退与性能回归判断 |
| Chapter09 | 09.9 | 成熟框架 Benchmark/Profiling 对照 |
| Chapter10 | 10.1 | ServiceCore 与 Runtime 的调用边界 |
| Chapter10 | 10.2 | Admission、容量、排队和背压 |
| Chapter10 | 10.3 | 取消、Deadline、失败和统一资源释放 |
| Chapter10 | 10.4 | Transport、网络库和 HTTP 映射 |
| Chapter10 | 10.5 | 同步 Completion 请求闭环 |
| Chapter10 | 10.6 | SSE、断连和写回背压 |
| Chapter10 | 10.7 | 请求并发、Worker 和上下文隔离 |
| Chapter10 | 10.8 | Metrics、错误映射和可观测性 |
| Chapter10 | 10.9 | 真实模型服务负载与可靠性验证 |
| Chapter10 | 10.10 | 成熟 Serving 框架对照 |

## 旧正文迁移

| 现有资产 | 目标位置 | 处理方式 |
| --- | --- | --- |
| Chapter06 `01`–`03` | 06.1–06.3、06.6 | 保留输入组装和单请求循环，删除多请求暗示 |
| Chapter06 `04_kv_cache.md` | 06.6、07.1–07.3 | 单请求增长语义留在 Chapter06，存储和寻址进入 Chapter07 |
| Chapter06 `05_scheduler.md` | 08.1–08.2 | 整体迁移到调度章 |
| Chapter06 `06_sampler.md` | 06.5 | 保留并补停止条件 |
| Chapter06 `07_end_to_end_generation.md` | 06.7–06.8 | 拆为流式契约和真实模型闭环 |
| Chapter06 `08_vllm_comparison.md` | 06.9、07.10、08.6 | 按 Generation、Cache、Scheduler 三类对象拆开 |
| Chapter06 `09_request_lifecycle.md` | 06.4、10.2–10.3 | 模型请求状态留在 Chapter06，服务接入与释放进入 Chapter10 |
| Chapter06 `10_continuous_batching.md` | 08.2–08.4 | 整体迁移到调度章 |
| Chapter06 `11_real_model_batch.md` | 08.4 | 保留真实 Tensor batch 和事务证据 |
| Chapter06 `12_block_kv_cache.md` | 07.3–07.7 | 整体迁移到 KV 章 |
| Chapter07 全部正文 | Chapter09 `09.1`–`09.4`、`09.8`–`09.9` | 与 Profiling 合并为同一证据闭环 |
| Chapter08 全部正文 | Chapter09 `09.5`–`09.9` | 保留工具、热点、A/B 和真实服务归因 |
| Chapter09 全部正文 | Chapter10 | 服务 benchmark 方法回链 Chapter09，服务结果留在 10.9 |

## 代码与测试迁移

| 当前对象 | 目标模块 | 必须保留的验证 |
| --- | --- | --- |
| `Request`、`GenerationConfig`、`Runtime`、`Sampler` | `code/chapter06_generation` | 请求状态、采样、单请求生成、失败路径 |
| `Embedding`、`PositionEncoding`、`Transformer`、`RequestModelExecutor` | `code/chapter06_generation` | 合成输入与真实 TinyLlama 单请求结果 |
| `KVCache` | Chapter06 只保留接口语义；实现迁入 `code/chapter07_kv_cache` | 连续缓存增长和请求隔离 |
| `CpuPagedKvCache` | `code/chapter07_kv_cache` | payload 写读、跨页寻址、Paged Attention 数值 |
| `BlockTable`、`KvBlockPool`、`BlockLocation` | `code/chapter07_kv_cache` | 容量、回收、事务和失败不泄漏 |
| `RoundRobinScheduler`、`TensorBatchScheduler` | `code/chapter08_scheduling` | 顺序、公平性和 Tensor batch shape |
| `StepBatchPlanner`、`ContinuousBatchScheduler` | `code/chapter08_scheduling` | token/request/block 预算、Chunked Prefill |
| `ExecutorBatchInput`、`RequestExecutionState` | `code/chapter08_scheduling` | Step plan 到执行输入和提交/回滚 |
| Chapter07 Benchmark 库、脚本和结果 | `code/chapter09_performance/benchmark` | 固定 workload、结果 schema、重复性检查 |
| Chapter08 Profiling 库、脚本和结果 | `code/chapter09_performance/profiling` | 阶段归因、A/B 决策和结果选择器 |
| `ServingCore`、HTTP Adapter、ResponseWriter、cpp-httplib Transport | `code/chapter10_serving` | admission、并发、取消、SSE、错误和指标 |

现有 `serving_preview.h/.cpp` 必须按对象拆分，不能整体重命名。Chapter07 不包含等待队列，Chapter08 不拥有 K/V payload，Chapter10 不实现 StepBatchPlan。

## 迁移顺序与验收门槛

| 阶段 | 修改范围 | 通过条件 |
| --- | --- | --- |
| 1 文档骨架 | 新建 Chapter07–10 目录和 intro/正文骨架，不进入导航 | 小节编号、职责和旧资产映射完整；旧页面仍可访问 |
| 2 Generation Core | 收紧 Chapter06 文档、代码和测试 | 单请求 L1/L2 通过；不再依赖 Scheduler 或 BlockPool |
| 3 KV Cache | 新建 Chapter07 代码、测试和正文 | 连续/分页 KV、事务、Paged Attention 独立通过 |
| 4 Scheduling | 新建 Chapter08 代码、测试和正文 | Scheduler 只通过 Chapter07 公共接口管理缓存 |
| 5 Performance | 合并原 Chapter07/08 | Benchmark 与 Profiling 共用 workload ID 和结果 schema |
| 6 Serving | 迁移原 Chapter09 到 Chapter10 | ServiceCore 通过 Chapter08 接口提交请求，Transport 契约保持不变 |
| 7 切换与清理 | 更新 README、USAGE、VitePress、Reference 和 CI | 无旧链接；默认回归与 Release 契约通过后删除旧目录 |

每个阶段按“正文边界 -> 图 -> 代码 -> 测试 -> 文档/代码核对”完成。迁移期间不修改旧页面内容；目标页面成为唯一编辑源后，才允许切换导航。

## 当前迁移状态

| 范围 | 已完成 | 后续扩展 |
| --- | --- | --- |
| Chapter06 | `next/` 目标正文、Generation Core 主图、词元回调契约、独立验证入口；调度和分页实现已移出库目标 | 异步模型执行 |
| Chapter07 | `next/` 07.1–07.10、分页数据面、真实 TinyLlama Decoder 直读和连续/分页 golden | Prefix Cache、GPU 分页内核 |
| Chapter08 | `next/` 08.1–08.6、优先级老化、真实 Decoder batch、分页事务和 Overlap 控制面 | 协作式抢占和设备事件重叠 |
| Chapter09 | `next/` 09.1–09.9、统一证据 schema、旧格式导入器和四组真实服务基线 | 更多 workload 与 GPU 性能证据 |
| Chapter10 | `next/` 10.1–10.10、优先级 admission、真实分页 batch、请求级 Worker、HTTP/SSE 和资源释放 | 自动 Deadline、异步背压恢复 |

全局导航、Reference、CI 和多模态服务依赖已经切换到 Chapter06–10 新结构。旧 Chapter07 Benchmark 与 Chapter08 Profiling 资产归档到 Chapter09 `legacy/`，旧 Chapter09 Serving 实现和重复教程页面已经删除；默认回归、Release 契约、站点构建和多模态 Chapter10 集成构建均已通过。

## 重构前置修改

文档样式和结构检查已经改为自动发现 `source/chapterNN_*`，Chapter10 将沿用现有章节规则。创建 Chapter10 学习者正文时，还要同时增加 `code/chapter10_serving/README.md`、`docs/code/chapter10_serving.md` 和章节主图；全部目标章节完成后再更新 VitePress 侧边栏与引用资产清单。
