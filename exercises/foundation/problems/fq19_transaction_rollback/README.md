# FQ19 Transaction Rollback

## 任务

实现一个最小的双槽位事务：先预留两个槽位，依次写入，再一起提交。通过故障开关模拟第一次写入后的失败，验证回滚会清除数据并归还全部资源。

## 完成标准

- 预留前先检查索引、重复槽位和空闲状态。
- 中途失败后，两个槽位都回到空闲状态且旧数据被清除。
- 成功路径只在两次写入完成后提交。
- 对已占用槽位的写入失败时不改变其他资源。
- 测试同时检查返回值、状态数量和槽位内容。

## 验证

复制并完成 starter 后，先运行自己的答案：

```bash
cp -r exercises/foundation/problems/fq19_transaction_rollback/starter /tmp/fq19-answer
bash scripts/workflows/check_foundation_answer.sh /tmp/fq19-answer/main.cpp
```

运行 `bash exercises/foundation/run_all.sh`，重点检查 `fq19_transaction_rollback` 的故障路径、成功提交和重复释放。

## 提示

- L0：把槽位区分为 `Free`、`Reserved` 和 `Committed` 三种状态。
- L1：失败发生后按与写入相反的方向恢复状态和数据。
- L2：测试故障是否发生只是第一步，还要验证空闲数、预留数和已提交数。

## 主线映射

对应 [`Foundation 调试与测试`](/foundation/04_debugging_testing_basics)、[`Chapter07 分页事务`](/chapter07_kv_cache/next/03_paged_transaction) 和 [`Chapter10 终态释放`](/chapter10_serving/next/03_terminal_release)。
