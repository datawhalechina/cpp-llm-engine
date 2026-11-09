# 07.5 把真实模型迁移到分页缓存

`RealLlamaCpuContext` 默认保留连续 K/V 作为单请求参考。Engine 可以为 Context 绑定 `RealLlamaKvDataPlane`，把真实 Decoder 的写入与 Attention 读取切换到 `CpuPagedKvCache`。权重、RoPE、Attention 公式和 Generation Runtime 契约保持不变。

## 迁移需要改动哪些连接

Context 先绑定抽象 K/V 数据面，Engine 再适配请求 ID 和分页缓存。每层 Q/K/V 投影完成后写入预留槽位，全部层写完才提交；Attention 根据页表和有效词元数读取历史。请求进入终态后统一释放物理块。

| 迁移步骤 | 必须保持的证据 |
| --- | --- |
| Context 绑定分页数据面 | 两个请求的状态仍隔离 |
| 每层写入预留槽位 | 所有层写完才能提交 |
| Attention 读取页表 | 单层输出与连续基线一致 |
| 22 层 Decoder 使用分页地址 | logits 误差在门槛内 |
| Generation Runtime 接入事务 | 固定 greedy token 不变 |
| 请求终态释放物理块 | 重复请求无泄漏、无串扰 |

CUDA Context 对连续和分页数据面使用同一组模型权重与批量 Decoder 算子。连续 Prefill 把 K/V 追加到请求独占缓存；分页 Prefill 根据请求 BlockTable 把同一个 chunk 写入共享物理 payload，并让多 Query Attention 直接读取页表。两条路径都只在完整 chunk 成功后推进 Context 位置。

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

真实 CUDA 分页 Prefill 的逐 Token/Tensor A/B 登记为 [`EXP-GPU-05`](../../reference/experiment_assets.md#exp-gpu-05)。固定 32 Token 结果支持使用 Tensor 路径，但性能结论只适用于登记的设备和工作负载。
