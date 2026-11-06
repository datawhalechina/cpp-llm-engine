# 07.9 把真实模型从连续缓存迁到分页缓存

`RealLlamaCpuContext` 默认保留连续 K/V，作为单请求基线；Engine 可以为 Context 绑定 `RealLlamaKvDataPlane`，把真实 Decoder 的写入与 Attention 读取切换到 `CpuPagedKvCache`。权重、RoPE、Attention 公式和 Generation Core 契约不随存储实现变化。

## 迁移链路

| 步骤 | 修改对象 | 必须保持的证据 |
| --- | --- | --- |
| 1 | Context 绑定抽象 K/V 数据面，Engine 适配请求 ID 和分页缓存 | 两个请求的状态仍隔离 |
| 2 | 每层 Q/K/V 投影后，把新 K/V 写到预留槽位 | 每层写入完整后才能提交 |
| 3 | Attention 接收 BlockTable 和有效词元数 | 单层输出与连续基线一致 |
| 4 | 22 层 Decoder 全部使用分页地址 | logits 误差在门槛内 |
| 5 | Generation Runtime 接入缓存事务 | 固定 greedy token 不变 |
| 6 | 请求终态统一释放物理块 | 重复请求无泄漏、无串扰 |

连续路径仍用于单请求参考；ServiceCore 在 admission 时把真实 Executor 绑定到分页数据面。`real_tinyllama_batch_smoke` 对同一 TinyLlama 输入分别执行连续和分页路径，并比较 logits、greedy token、位置拒绝与失败回滚。

## 为什么不能只比较最终文本

最终文本相同不能定位误差来自哪一层。核对应按从小到大推进：单层 Attention、单层 Decoder、22 层 logits、固定词元序列、重复请求资源释放。前一层证据通过后，才扩大验证范围。

| 核对层级 | 比较对象 | 失败时优先检查 |
| --- | --- | --- |
| 单层 | Attention 输出张量 | 页表地址、GQA 头映射、有效长度 |
| 多层 | Decoder hidden state | 层号、提交顺序、RoPE 位置 |
| 输出层 | logits 和 argmax | 全层累积误差、最后位置选择 |
| 请求级 | 生成词元与终态 | Prefill/Decode 计数、回滚与释放 |

## 当前验证入口

`chapter07_paged_kv_cache_test` 验证分页数据面的跨块寻址、逐层提交和复用隔离。运行 `bash code/chapter07_kv_cache/scripts/run_real_paged_kv.sh` 后，`real_tinyllama_paged_kv_test` 验证 22 层 Decoder 的 Attention 直接读取分页 payload，并与连续 Context 得到相同 logits 和固定词元，同时检查错误位置、非法词元、回滚和释放。Chapter08、Chapter10 再分别验证批量执行与服务终态。GPU PagedAttention 与跨请求 Prefix Cache 仍是后续扩展。
