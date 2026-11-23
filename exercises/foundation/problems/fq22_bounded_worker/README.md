# FQ22 Bounded Worker Lifecycle

## 任务

实现一个单线程有界 Worker。请求先进入容量受限的队列，Worker 使用条件变量等待任务；待处理请求可以取消，任务异常转成失败终态，停止时排空已接收任务并安全回收线程。

## 完成标准

- 容量满、重复请求编号和停止后的提交被拒绝。
- Worker 使用 `std::thread`、`std::mutex` 和 `std::condition_variable` 等待任务。
- 待处理请求可以取消，取消后释放队列容量。
- 任务异常转换为失败终态，不逃出 Worker 线程。
- 每个已接收请求只产生一个完成、失败或取消终态。
- Worker 启动后，`stop()` 排空已接收任务、唤醒线程，并安全 `join()`。

## 验证

复制并完成 starter 后，先运行自己的答案：

```bash
cp -r exercises/foundation/problems/fq22_bounded_worker/starter /tmp/fq22-answer
bash scripts/check_foundation_answer.sh /tmp/fq22-answer/main.cpp
```

运行 `bash exercises/foundation/run_all.sh`，重点检查 `fq22_bounded_worker` 的容量、取消、异常、停止和终态唯一性。

## 提示

- L0：所有队列、终态和停止标志都由同一互斥锁保护。
- L1：条件变量谓词是“正在停止或队列非空”，防止虚假唤醒破坏状态。
- L2：执行任务时释放互斥锁，记录终态时再加锁；析构函数调用幂等的 `stop()`。

## 选做：有界等待与超时

增加 `wait_for_terminals_for(count, timeout)`：使用 `std::condition_variable::wait_for` 等待终态数量，满足条件返回 `true`，超时返回 `false`。谓词仍然检查 `completions.size() >= count`，不能把一次唤醒当作完成。

至少验证两个场景：已有终态时立即成功；没有足够终态时在有限时间内返回 `false`。这个超时只约束调用者等待结果的时间，不应强行终止正在执行的 C++ 函数。Chapter10 的请求截止时间还需要把取消意图传入请求状态，并在安全点释放调度和 K/V 资源。

## 主线映射

对应 [`Foundation 并发基础`](/foundation/10_concurrency_basics)、[`Chapter10 请求 Worker`](/chapter10_serving/next/07_concurrency_workers) 和 [`终态释放`](/chapter10_serving/next/03_terminal_release)。
