# 07.5 把真实模型迁移到分页缓存

`RealLlamaCpuContext` 默认保留连续 K/V 作为单请求参考。引擎适配层可以为 Context 绑定 `RealLlamaKvDataPlane`，把真实 Decoder 的写入与 Attention 读取切换到 `CpuPagedKvCache`。权重、RoPE、Attention 公式和 Generation Runtime 接口保持不变。

## 迁移需要改动哪些连接

Context 先绑定抽象 K/V 数据面，引擎适配层再连接请求 ID 和分页缓存。每层 Q/K/V 投影完成后写入预留槽位，全部层写完才提交；Attention 根据页表和有效词元数读取历史。请求进入终态后统一释放物理块。

| 迁移步骤 | 应检查的结果 |
| --- | --- |
| Context 绑定分页数据面 | 两个请求的状态仍隔离 |
| 每层写入预留槽位 | 所有层写完才能提交 |
| Attention 读取页表 | 单层输出与连续基线一致 |
| 22 层 Decoder 使用分页地址 | logits 误差在门槛内 |
| Generation Runtime 接入事务 | 固定 greedy token 不变 |
| 请求终态释放物理块 | 重复请求无泄漏、无串扰 |

CUDA 上下文对连续和分页数据面使用同一组模型权重与批量解码器算子。连续预填充把 K/V 追加到请求独占缓存；分页预填充根据请求块表把同一个分块写入共享物理载荷，并让多查询注意力直接读取页表。两条路径都只在完整分块成功后推进上下文位置。

## 模型架构怎样改变分页适配

块分配、引用计数和提交回滚可以保持平台无关，真正保存的载荷仍由注意力架构决定。当前真实分页测试以 TinyLlama 的标准 GQA K/V 为基线；其他模型需要先确认自己的缓存表示能否直接放入相同布局。

| 模型 | 请求级 Attention 历史 | 迁移到 Chapter07 时的处理 |
| --- | --- | --- |
| TinyLlama | 每层标准 key/value，多个查询头共享 K/V 头 | 已接入标准分页载荷 |
| Qwen3 | 标准 GQA K/V，头数和 RoPE 参数不同 | 复用载荷形状，另建模型级分页固定结果 |
| Qwen1.5-MoE | 注意力仍使用标准 K/V，MoE 只改变前馈网络 | 可复用标准载荷，当前执行器工厂尚未绑定数据面 |
| DeepSeek-V2-Lite | 每层保存低秩 latent K/V 与共享位置 Key | CPU 分页组件已固定载荷和事务，CPU Factory 仍使用连续 Context；CUDA 已完成直接页表读取、真实 Context 绑定和 27 层验收 |

DeepSeek 的压缩缓存原理见 [`04.1 多头注意力与 MLA`](../../chapter04_attention/01_multihead_attention.md)。这里不把 latent 强行展开为标准 K/V 再长期保存，因为那会丢失 MLA 降低缓存容量的设计目标。

`chapter07_paged_mla_cache_test` 验证跨块顺序、逐层提交、批量原子性和物理块复用清理。CUDA 专项测试再使用非连续页表比较连续与分页 MLA Attention。`RealDeepSeek2CudaExecutorFactory` 把请求 ID、共享块池、设备压缩载荷和 Context 连接起来；真实模型验收已比较连续与分页的 27 层 logits，并验证双请求批提交和终态资源计数。

```bash
CUDA_HOME=/usr/local/cuda-12.8 bash code/cuda/scripts/run.sh
CUDA_HOME=/usr/local/cuda-12.8 \
  bash code/engine/scripts/run_deepseek_v2_lite_cuda_runtime.sh
```

第二个入口通过时应输出分页 logits 误差、`paged_mla_batch: true` 和 `paged_mla_terminal_release: true`。这些字段分别证明数值等价、跨请求事务和最终物理块回收。

[`EXP-GPU-18`](../../reference/experiment_assets.md#exp-gpu-18) 使用同一 DeepSeek-V2-Lite 模型比较连续与分页 MLA。短上下文结果显示两条路径成本处于同一量级，连续与分页 logits 一致。分页路径在这里主要解决按需分块、跨请求复用和终态回收；具体样本与设备环境统一查阅实验资产。

## 为什么要逐级比较

只比较最终文本无法定位误差。验证应从单层 Attention 开始，再扩大到 Decoder hidden state、22 层 logits、固定词元序列和重复请求资源释放。前一级成立后，后一级失败才有明确排查范围。

| 核对层级 | 比较对象 | 失败时优先检查 |
| --- | --- | --- |
| 单层 | 多 Query Attention 输出 | 页表地址、GQA 映射、因果可见范围 |
| 多层 | Decoder hidden state | 层号、提交顺序、RoPE 位置 |
| 输出层 | logits 和 argmax | 累积误差、最后位置选择 |
| 请求级 | 生成词元与终态 | Prefill/Decode 计数、回滚与释放 |

## 验证 TinyLlama 分页路径

脚本构建真实模型测试，对同一输入分别运行连续和分页路径。通过条件是 22 层多 Token Prefill 和 Decode 都直接读取分页数据，连续与分页 logits 和固定词元一致，同时跨页、错误位置、非法词元、回滚与释放都被检查。

```bash
bash code/chapter07_kv_cache/scripts/run_real_paged_kv.sh
```

真实 CUDA 分页预填充的逐词元/张量 A/B 登记为 [`EXP-GPU-05`](../../reference/experiment_assets.md#exp-gpu-05)。固定 32 词元结果支持使用张量路径，但性能结论只适用于登记的设备和负载。
