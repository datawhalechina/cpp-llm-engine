# Chapter04–08 测试所有权审核

本审核使用 `code/testing/chapter_test_contracts.json` 判断测试属于哪一章，不再用测试文件所在目录代替职责归属。

## 审核规则

| 证据层级 | 回答的问题 | 依赖规则 |
| --- | --- | --- |
| unit | 一个最小函数或状态转换是否正确 | 只依赖当前章或公共 Core |
| component | 本章多个对象能否形成局部闭环 | 只依赖当前章与前置章节 |
| contract | 公开状态、失败和所有权不变量是否成立 | 由定义契约的章节拥有 |
| integration | 前置章节产物能否被当前章消费 | 由后面的消费者章节拥有 |
| real_model | 真实模型是否通过本章接口 | 可以依赖前置章节，不得依赖后续章节 |
| e2e | 多章系统是否形成纵向闭环 | 放在 Engine 或验收层，不反向算作前置章节完成条件 |

## Chapter04

职责是 Decoder 数学、结构、Tensor shape 和 Model/Context 所有权。

| 有效证据 | 层级 | 结论 |
| --- | --- | --- |
| `chapter04_attention_test` | component | Attention、连续 K/V 数值基线、归一化、FFN、残差和 DecoderLayer 成立 |
| `chapter02_tensor_attention_integration_test` | integration | Chapter02 View 能进入 Attention，3D/4D 与失败保持成立 |

Chapter04 不再把量化 TinyLlama token golden 作为本章完成条件。当前 L2 使用 `gguf_first_layer_inspect` 检查真实配置与权重映射，但这仍是手动结构证据；后续应增加不依赖 Chapter05 的 `chapter04_real_structure_contract_test`，固定 TinyLlama 的 22 层、hidden size、GQA、RoPE、RMSNorm 和九个首层权重映射。

## Chapter05

职责是量化表示、packed block、QuantizedLinear Backend、误差和注入 Decoder 后的真实数值。

| 有效证据 | 层级 | 结论 |
| --- | --- | --- |
| `chapter05_quantization_test` | component | INT8、Q4_K、Q6_K、Q8_0、reference、fused、policy 和错误路径成立 |
| `chapter02_tensor_quantized_linear_integration_test` | integration | 公共 Tensor View 能进入量化 Linear |
| `gguf_decoder_multilayer` | real_model | 量化后端进入 22 层 TinyLlama Decoder，并形成 logits 数值对照 |

真实 Decoder 已注册为独立 CTest，由 `run_real_decoder.sh` 单独运行。Generation、batch 和 ServiceCore 性能不再计入 Chapter05 完成条件。

## Chapter06

职责是单请求 Request、Prefill/Decode、Sampler、停止、回调和终态。

| 有效证据 | 层级 | 结论 |
| --- | --- | --- |
| `chapter06_generation_test` | component | 单请求 Prefill、Decode、上下文限制、EOS 和回调顺序成立 |
| `chapter06_sampler_test` | unit | Greedy 选择成立 |
| `chapter06_kv_cache_test` | unit | 当前 token history 的追加与清理成立；对象命名仍需修正 |
| `chapter06_request_contract_test` | contract | 状态机和终态吸收成立 |
| `real_tinyllama_generation_smoke` | real_model | Chapter04/05 Executor 能被 Runtime 驱动并得到固定 token |

`run_generation_core.sh` 只构建上述默认核心测试；`run_real_generation.sh` 只运行真实单请求 Generation。

## Chapter07

职责是 BlockTable、物理块池和分页 K/V payload，不决定请求执行顺序。

| 有效证据 | 层级 | 结论 |
| --- | --- | --- |
| `chapter07_block_pool_test` | contract | 容量守恒、块独占、提交单调、失败原子性和释放复用成立 |
| `chapter07_paged_kv_cache_test` | component | 分层写入、跨块 Attention 数值和 payload 复用隔离成立 |

## Chapter08

职责是请求选择、预算、公平性、Continuous Batching、Tensor Batch 和 Chapter07 资源事务编排。

| 有效证据 | 层级 | 结论 |
| --- | --- | --- |
| `chapter08_scheduler_test` | component | Round-robin、终态跳过、流式归属和逐请求执行器隔离成立 |
| `chapter08_step_scheduler_test` | component | Token 预算、Chunked Prefill、公平推进、块压力延迟准入和三层状态一致性成立 |
| `chapter08_cache_transaction_test` | integration | Decode 写槽预留和执行失败回滚成立 |
| `chapter08_tensor_batch_test` | component | Ragged 输入、批量 Prefill、重复请求原子性和逐请求准入成立 |

原先位于 `code/chapter06_generation/tests` 的 9 个缓存与调度测试已完成行为迁移并删除。迁移不是文件改名：Runtime 失败传播留在 Chapter06，资源不变量进入 Chapter07，策略与跨章事务进入 Chapter08。

## 结论与优先级

1. Chapter04–08 的测试文件均已登记 active owner、证据层级和前置依赖，没有 legacy 测试债务。
2. CTest 标签可按 `owner-chapterNN`、`level-*` 和 `status-active` 选择测试；`unit`、`component` 等词不再作为自由标签重复登记，避免层级冲突。
3. Chapter06 源码仍编译历史 Scheduler 与 Paged K/V 实现；测试迁移完成不等于实现物理拆分完成。
4. Chapter04 的真实结构契约仍是待补证据；当前只有手动 `gguf_first_layer_inspect` 入口。
5. Chapter06 的教学 `KVCache` 仍应重命名为 token history，避免与 Chapter07 模型 K/V 混淆。
6. 下一轮将同一清单规则扩展到 Chapter00–03 和 Chapter09–10，最终覆盖整条主线。

