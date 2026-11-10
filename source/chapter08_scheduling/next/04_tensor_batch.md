# 08.4 把调度计划交给真实解码批次

步骤批次（Step batch）回答本步选择哪些请求，执行器批次（Executor batch）补齐词元、位置和 K/V 写入位置，张量批次（Tensor batch）再把多行激活交给共享模型算子。真正的模型批量计算会在一次共享算子调用中处理多行激活；循环调用单请求执行器仍是顺序执行。

## 三层批次分别保存什么

| 层级 | 主要数据 | 产物 |
| --- | --- | --- |
| 步骤批次（Step batch） | 请求 ID、阶段、词元数、预算 | 候选请求列表 |
| 执行器批次（Executor batch） | 词元、位置、K/V 写入槽 | 可校验的模型输入行 |
| 张量批次（Tensor batch） | 多行激活、行边界或批次维度 | 一次算子或解码器调用的多行输出 |

## 变长请求不要求先补齐

当前教学张量批次保存每行边界，真实解码批次则为每个请求携带独立位置和块表（BlockTable），因此不同历史长度可以进入同一步。某些稠密批量实现会用填充词元（PAD）补成矩形序列；填充只改变布局，有效词元、位置和 K/V 长度保持不变。输入超过 `max_context_tokens` 时，Chapter06 会在执行前报告错误，确保提示词不会被静默截断。

## 当前真实 Decode 合并了哪些计算

`RealLlamaCpuModel::decode_batch()` 让多个请求共享只读 TinyLlama 权重，每个请求保留独立上下文（Context）。解码阶段的 Q/K/V、输出投影、FFN 和语言模型头（LM Head）使用多行线性计算；不同历史长度的注意力仍按请求 ID 和块表分别读取分页 K/V。当前真实多请求预填充（Prefill）逐请求执行，张量批量路径先覆盖解码阶段。

| 执行部分 | 当前方式 | 还需要什么 |
| --- | --- | --- |
| 量化 Linear | CPU 多行共同遍历权重；CUDA 用二维网格直接读取 packed 权重 | AVX2 专用批量内核和 CUDA 权重复用调优 |
| Attention | 每个请求直接读取自己的分页 K/V | 融合的 Ragged/Paged Attention 批量内核 |
| Prefill | Runtime 支持分块，但真实模型仍逐请求执行 | 真实多请求 Prefill Tensor batch |
| 解码 | CPU 与 CUDA 均有真实多上下文批量调用 | 融合计算核、减少行打包和跨批量大小的容量复用 |

## 为什么批量临时张量需要复用

CUDA Decode 每一步都会使用批量 hidden、logits、embedding rows、stream、cuBLAS handle 和 Decoder 中间 Tensor。这些对象只服务本次计算，不保存请求历史；如果每一步重新创建，就会在生成循环中重复执行设备分配与释放。

当前 CUDA 执行器工厂（`RealLlamaCudaExecutorFactory`）按精确批量大小保存一个持久化工作区（workspace）。例如，批量大小为 2 的连续解码步骤会复用同一组临时资源；批量大小变为 3 时单独创建另一组，避免无效填充行参与计算。工作区创建时按模型最大上下文预留 Attention score 和页表空间，位置增长不会在每个 Decode step 触发设备分配。量化 Linear 在同一次二维 kernel 网格中处理这些输入行，不再为每次调用展开完整 Float32 权重。互斥锁保证同一工作区串行复用；每个请求的位置和 K/V 绑定继续保存在各自的模型执行上下文中。多批次并发需要进一步拆分工作区和执行流。

## 分页 K/V 怎样提交

ServiceCore 在 admission 时把请求 ID 绑定到真实 Executor 的分页数据面。进入批量 Decode 前，Scheduler 为每个候选预留一个槽位并构造 token、position、request ID 和 K/V 写入位置；K/V 位置用于批输入校验，真实 Runner 再通过 request ID 找到已绑定的数据面。Decoder 逐层写入 K/V，Attention 通过页表读取历史和当前行。

成功时 Runner 必须让每个模型 Context 恰好推进一个位置，并返回同顺序 logits；失败时 Runner 负责恢复模型 Context。Scheduler 随后负责提交或回滚 Request 与 Chapter07 资源，因为它无法直接修改模型内部状态。

```text
admission bind request ID
  -> per-request prefill writes paged payload
  -> reserve one slot per request
  -> build ExecutorBatchInput
  -> real decode_batch
  -> every layer reads/writes paged K/V
  -> commit the complete batch
```

| 当前证据 | 后续扩展 |
| --- | --- |
| 顺序与 batch logits、greedy token 一致 | 融合的 CPU PagedAttention 内核 |
| 错误 position 与无效 token 会整体回滚 | CUDA 双请求分页 batch 已接入同一事务边界 |
| CUDA 连续两步复用同尺寸 workspace | 多 batch 并发与容量型 workspace |
| ServiceCore 自动传递 request ID 并选择真实 batch | 协作式抢占和设备事件重叠 |
| Context position、页表和分页 payload 按整批提交 | 跨请求 Prefix Cache |

[CUDA 双请求批执行实验](../../reference/experiment_assets.md#exp-gpu-06)在真实 TinyLlama、分页 K/V 和一次 Decode 下比较顺序执行与 Tensor Batch。两轮 batch/顺序中位数比例为 `0.922/0.919`，逐请求 logits 完全一致；这说明当前小批量融合路径减少了重复执行成本，但尚未回答更大批量和长期服务吞吐。

Qwen1.5-MoE 还需要先按 Router 结果把 token 归入专家组。[MoE 固定队列实验](../../reference/experiment_assets.md#exp-cpu-17)使用四个请求，在 Prefill 后连续执行三个 Chapter08 调度步；按专家分组与逐行执行的中位数比例为 `0.582`，完整生成序列和每步批量调用计数一致。该结果说明调度层能够持续使用分组执行路径。Chapter08 用固定队列隔离调度行为；[Chapter10 动态准入实验](../../reference/experiment_assets.md#exp-cpu-18)再在两个请求运行期间加入另外两个请求，验证输出和终态释放。

下图强调共享模型计算与独立请求历史的分界：Linear 和 LM Head 接收多行激活，每个请求的 Attention 仍通过自己的 BlockTable 访问分页 K/V。

![从调度计划到真实 Tensor Batch](../../assets/diagrams/chapter08-real-batch-execution.svg)

## 验证教学批次和真实模型

```bash
bash code/chapter08_scheduling/scripts/run.sh
bash code/chapter08_scheduling/scripts/run_real_batch.sh
```

| 环境 | 必须通过 |
| --- | --- |
| 无模型文件 | Ragged batch、重复请求或槽位拒绝、缓存事务和结果等价 |
| 本地 TinyLlama | 真实 batch golden、批量 Linear、失败回滚和分页计数一致 |
| 本地 Qwen1.5-MoE | 独立 K/V、跨请求专家分组、连续批量调用和完整生成序列一致 |
