# 06.4 推进请求状态并保持终态

生成循环必须区分“暂时没有继续执行”和“请求已经结束”。请求阶段（`RequestPhase`）把正常完成、外部取消、超时、准入拒绝和模型失败保存为不同结果，调用方可以据此决定是否继续推进以及怎样报告结束原因。

## 状态怎样变化

请求从 `Initial` 开始。完整 prompt 提交后进入 `Prefilled`，开始逐词元推进后进入 `Decoding`。正常停止进入 `Finished`；其余五种终态记录不同的结束来源。

| 当前阶段 | 事件 | 下一阶段 | 可观察结果 |
| --- | --- | --- | --- |
| `Initial` | 一段 Prefill 成功但 prompt 未结束 | `Initial` | `prefill_offset` 增加 |
| `Initial` | 完整 Prefill 成功 | `Prefilled` | prompt 已全部提交 |
| `Prefilled` | 开始提交生成词元 | `Decoding` | `generated_size` 增加 |
| `Prefilled` / `Decoding` | EOS 或输出长度上限 | `Finished` | 正常结束 |
| 任意非终态 | 外部取消 | `Cancelled` | 后续调用不再推进 |
| 任意非终态 | Deadline 到达 | `TimedOut` | 后续调用不再推进 |
| `Initial` | 上下文预算不满足 | `Rejected` | 不调用模型 |
| 任意非终态 | 模型调用失败 | `Failed` | `failure_reason` 保存原因 |

状态实现位于 <a href="/_code/chapter06_generation/src/request.cpp"><code>request.cpp</code></a>。`terminal()` 将五种结束状态统一为“不可继续执行”，而 `finished()` 只表示正常完成。

## 为什么终态必须吸收后续事件

终态请求可能仍被旧队列、回调或调用者持有。如果 `cancel()` 后还能调用 `mark_decoding()`，同一请求就可能再次进入模型并增加输出。因此 `append_generated_token()`、`advance_prefill()` 和所有 `mark_*()` 都先检查终态。

| 终态后再次调用 | 预期行为 |
| --- | --- |
| 追加生成词元 | 忽略，输出长度不变 |
| 增加 Prefill 游标 | 忽略，位置不变 |
| 标记 Prefilled/Decoding/Finished | 保持原终态 |
| 再次取消、超时、拒绝或失败 | 保持首次终态和首次失败原因 |

状态不可恢复不等于资源已经释放。请求专属模型 Context 和物理 KV Block 的释放分别由其所有者执行；Chapter06 只保证终态对象不会再次产生模型工作。

## 验证终态契约

```bash
cmake -S code/chapter06_generation -B /tmp/cpp-llm-chapter06-state \
  -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/cpp-llm-chapter06-state \
  --target chapter06_request_contract_test -j2
ctest --test-dir /tmp/cpp-llm-chapter06-state \
  -R '^chapter06_request_contract_test$' --output-on-failure
```

测试以表驱动方式遍历 `Finished`、`Cancelled`、`TimedOut`、`Rejected` 和 `Failed`，再尝试全部状态转换。通过条件是阶段、Prefill 游标、生成词元和失败原因均保持不变。下一篇使用正常终态定义采样与停止条件。
