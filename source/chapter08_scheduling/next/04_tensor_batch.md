# 08.4 把调度计划交给真实批量执行

Step batch 回答本步选择谁，Executor batch 补齐 token、position 和 K/V 写入位置，Tensor batch 再把多行激活交给共享模型算子。循环调用多个单请求 Executor 不属于真正的模型批量计算。

## 三层批次分别保存什么

| 层级 | 主要数据 | 产物 |
| --- | --- | --- |
| Step batch | request ID、阶段、token count、预算 | 候选请求列表 |
| Executor batch | token、position、K/V 写入槽 | 可校验的模型输入行 |
| Tensor batch | 多行激活、行边界或 batch 维度 | 一次算子或 Decoder 调用的多行输出 |

## 当前真实 Decoder 合并了哪些计算

`RealLlamaCpuModel::decode_batch()` 让多个请求共享只读 TinyLlama 权重，每个请求保留独立 Context。Q/K/V、输出投影、FFN 和 LM Head 使用批量 Linear；不同历史长度的 Attention 按请求 ID 和 BlockTable 读取各自分页 K/V。

| 执行部分 | 当前方式 | 还需要什么 |
| --- | --- | --- |
| 量化 Linear | 多行激活共同遍历权重 | AVX2 专用批量内核和更少打包开销 |
| Attention | 每个请求直接读取自己的分页 K/V | 融合的 Ragged/Paged Attention 批量内核 |
| Prefill | Runtime 支持分块 | 真实多请求 Prefill Tensor batch |
| Decode | 真实多 Context 批量调用 | 减少打包开销并接入 GPU backend |

## 分页 K/V 怎样提交

ServiceCore 在 admission 时把请求 ID 绑定到真实 Executor 的分页数据面。Prefill 和 Decode 都先预留槽位，Decoder 逐层写入 K/V，Attention 通过页表读取历史和当前行；所有层完成后提交，任一步失败都裁剪到执行前的位置。

```text
admission bind request ID
  -> prefill writes paged payload
  -> reserve one slot per request
  -> real decode_batch
  -> every layer reads/writes paged K/V
  -> commit the complete batch
```

| 当前证据 | 后续扩展 |
| --- | --- |
| 顺序与 batch logits、greedy token 一致 | 融合的 CPU PagedAttention 内核 |
| 错误 position 与无效 token 会整体回滚 | GPU Tensor batch 和 PagedAttention backend |
| ServiceCore 自动传递 request ID 并选择真实 batch | 协作式抢占和设备事件重叠 |
| Context position、页表和分页 payload 同步提交 | 跨请求 Prefix Cache |

## 验证教学批次和真实模型

```bash
bash code/chapter08_scheduling/scripts/run.sh
bash code/chapter08_scheduling/scripts/run_real_batch.sh
```

| 环境 | 必须通过 |
| --- | --- |
| 无模型文件 | Ragged batch、重复请求拒绝、缓存事务和结果等价 |
| 本地 TinyLlama | 真实 batch golden、批量 Linear、失败回滚和分页计数一致 |
