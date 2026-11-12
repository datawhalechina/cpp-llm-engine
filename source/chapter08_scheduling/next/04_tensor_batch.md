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

不同模型共享 `ExecutorBatchRunner` 的整批成功或整批回滚规则，但批量执行内部不完全相同：

| 模型 | Attention 历史 | 跨请求复用 | 当前分页状态 |
| --- | --- | --- | --- |
| TinyLlama | 每请求标准 GQA K/V | Q/K/V、输出投影、FFN 与 LM Head 的多行 Linear | CPU/CUDA 已绑定 Chapter07 BlockTable |
| Qwen1.5-MoE | 每请求分层标准 K/V | 同层权重只加载一次，MoE 输入按专家重新分组 | Context 连续保存，尚未迁移到 Chapter07 |
| DeepSeek-V2-Lite CPU | 每请求低秩 MLA K/V 与共享位置 Key | 同层权重只加载一次，64 选 6 的 FFN 输入按专家分组 | MLA Context 连续保存，需要新的分页 payload 适配 |
| DeepSeek-V2-Lite CUDA | 每请求设备压缩 MLA K/V 与位置 Key | 同一专家的请求行共同执行 Gate/Up/Down，共享专家处理完整批次 | 设备 Context 连续保存，尚未迁移到 Chapter07 |

DeepSeek CUDA batch 保留了“Attention 隔离、FFN 合并”的边界。每个请求先在自己的执行流和压缩 MLA 历史上完成 Attention；所有请求到达 FFN 后，批量工作区读取 Router 结果，把命中同一专家的输入行打包，再调用 `quantized_linear_batch()` 完成 Gate、Up 和 Down。共享专家直接处理整个批次，结果按请求加权拆回。批量大小相同时，Factory 会复用同一个工作区，避免 26 个 MoE 层和后续 Decode step 重复分配临时张量。

这条路径已经执行真实跨请求量化计算，但仍保留三项清晰边界：Router、稳定 Top-K 和加权拆回在 Host 完成；最终归一化与 LM Head 按请求执行；压缩 MLA payload 尚未分页。它也不等于设备执行重叠或单个融合 grouped-MoE kernel。

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

| 当前证据 | 仍需扩展 |
| --- | --- |
| 顺序与 batch logits、greedy token 一致 | 融合的 CPU PagedAttention 内核 |
| 错误 position、无效 token 和 CUDA 双请求分页 batch 共用整批回滚边界 | 更大 batch 的故障注入与容量压力测试 |
| CUDA 连续两步复用同尺寸 workspace | 多 batch 并发与容量型 workspace |
| ServiceCore 自动传递 request ID 并选择真实 batch | 协作式抢占和设备事件重叠 |
| Context position、页表和分页 payload 按整批提交 | MoE/MLA Context 的分页 payload 适配 |

[CUDA 双请求批执行实验](../../reference/experiment_assets.md#exp-gpu-06)在真实 TinyLlama、分页 K/V 和一次 Decode 下比较顺序执行与 Tensor Batch。两轮 batch/顺序中位数比例为 `0.922/0.919`，逐请求 logits 完全一致；这说明当前小批量融合路径减少了重复执行成本，但尚未回答更大批量和长期服务吞吐。

Qwen1.5-MoE 还需要先按 Router 结果把 token 归入专家组。[MoE 固定队列实验](../../reference/experiment_assets.md#exp-cpu-17)使用四个请求，在 Prefill 后连续执行三个 Chapter08 调度步；按专家分组与逐行执行的中位数比例为 `0.582`，完整生成序列和每步批量调用计数一致。该结果说明调度层能够持续使用分组执行路径。Chapter08 用固定队列隔离调度行为；[Chapter10 动态准入实验](../../reference/experiment_assets.md#exp-cpu-18)再在两个请求运行期间加入另外两个请求，验证输出和终态释放。

DeepSeek-V2-Lite 的 batch 先按层加载一次权重，各请求使用自己的压缩 MLA 历史完成 Attention，再把该层 64 选 6 的 FFN 输入放进共同的专家分发表。真实 smoke 中两个请求在 Prefill 后通过一次 batch runner 调用同步 Decode，固定生成 `185 85770`，请求位置都从 1 推进到 2。它证明调度、Context 隔离、跨请求专家分组和整批回滚已经连通。[EXP-CPU-19](../../reference/experiment_assets.md#exp-cpu-19) 在 batch 2、单步 Decode 和全层缓存下记录到 `0.943` 的中位数比例；该结果只说明当前固定工作负载的有限收益。

下图强调共享模型计算与独立请求历史的分界：稠密模型让多行激活进入共同的 Linear，MoE 模型在 Attention 后按专家重新分组；每个请求仍从自己的标准 K/V 或压缩 MLA 历史计算 Attention。

![从调度计划到真实 Tensor Batch](../../assets/diagrams/chapter08-real-batch-execution.svg)

## 验证教学批次和真实模型

```bash
bash code/chapter08_scheduling/scripts/run.sh
bash code/chapter08_scheduling/scripts/run_real_batch.sh
bash code/chapter08_scheduling/scripts/run_deepseek_v2_lite_batch.sh
CUDA_HOME=/usr/local/cuda-12.8 \
bash code/engine/scripts/run_deepseek_v2_lite_cuda_batch.sh
```

| 环境 | 必须通过 |
| --- | --- |
| 无模型文件 | Ragged batch、重复请求或槽位拒绝、缓存事务和结果等价 |
| 本地 TinyLlama | 真实 batch golden、批量 Linear、失败回滚和分页计数一致 |
| 本地 Qwen1.5-MoE | 独立 K/V、跨请求专家分组、连续批量调用和完整生成序列一致 |
| 本地 DeepSeek-V2-Lite | 独立 MLA 历史、一次按层 batch 调用、固定词元与位置同步推进 |
| 本地 DeepSeek-V2-Lite + CUDA | 两个请求共享一个 batch workspace，26 层各执行一次 grouped MoE，重复 ID 失败时位置保持不变 |
