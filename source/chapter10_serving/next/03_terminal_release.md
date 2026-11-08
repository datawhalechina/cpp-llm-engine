# 10.3 让取消、超时和失败走同一释放路径

请求可能正常完成，也可能在 waiting、Prefill、Decode 或写回阶段停止。每种终态都要进入<a href="../../_code/chapter10_serving/src/service_core.cpp"><code>release_terminal()</code></a>，防止 Request、Executor、K/V block 或 Adapter 记录残留。

## 终态来源

| 终态 | 触发来源 | 必须停止什么 |
| --- | --- | --- |
| Finished | 停止词元或长度上限 | 后续调度与写词元 |
| Cancelled | 客户端断开或主动取消 | 后续模型执行 |
| TimedOut | Deadline 到期 | 排队或执行 |
| Failed | Executor、缓存事务或写回错误 | 当前事务和后续 step |

## 当前统一释放顺序

释放前先把请求标记为终态，避免它再次进入 StepBatchPlan。`release_terminal()` 清理 Request 的词元历史，释放 Chapter07 BlockTable，再移除服务条目；条目移除后，请求专属 Executor 引用随之释放。

| 顺序 | 操作 | 验证事实 |
| --- | --- | --- |
| 1 | 写入唯一终态 | 请求不再可调度 |
| 2 | 当前调度步完成提交或回滚 | Chapter08 测试固定事务结果 |
| 3 | 清理 Request 缓存并释放 BlockTable | block pool 恢复容量 |
| 4 | 移除服务条目和 Executor 引用 | inflight 归零 |
| 5 | 记录终态与 released 指标 | 重复调用不重复计数 |

当前测试覆盖 waiting 取消、running 完成、显式超时、Executor 失败、重复释放，以及分页资源随请求终态归还。模型执行中的抢占式取消需要异步 Executor 支持，属于后续并发执行扩展。

## 验证幂等释放

运行 `bash code/chapter10_serving/scripts/run.sh`。<a href="../../_code/chapter10_serving/tests/terminal_release_test.cpp"><code>chapter10_terminal_release_test</code></a> 检查取消、超时、Executor 失败、重复释放和分页块归还，最终 `inflight` 与已占用块数都回到预期值。`10.4` 开始把这组终态映射到网络连接。
