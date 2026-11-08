# F10 进程、线程与同步异步基础

推理服务同时涉及网络连接、请求状态、CPU Worker、GPU stream 和模型批次。它们都可能被称为“并发”，但共享的数据和完成信号不同。进入 Chapter08 和 Chapter10 前，先学会判断：代码在哪个执行单元运行，谁能同时访问状态，调用者何时得到结果。

## 进程与线程共享什么

进程拥有独立地址空间。两个进程默认不能直接读写同一份 C++ 对象，需要使用 IPC、共享内存或网络传递数据。线程位于同一进程内，可以共享模型权重和服务状态，但也可能同时修改同一容器，因此需要明确所有权和同步方式。

| 执行单元 | 默认共享 | 主要代价 | 教程中的位置 |
| --- | --- | --- | --- |
| 进程 | 不共享普通地址空间 | IPC、序列化、独立故障与内存 | Chapter10 成熟框架对照 |
| 线程 | 共享进程内存 | 数据竞争、锁和线程调度 | Chapter10 `RequestWorkerPool` |
| CUDA stream | 共享设备与显存，队列内有序 | Event、同步点和资源生命周期 | `code/cuda`、Chapter09 |

只读模型权重适合共享；Request、Context、Sampler 状态和 K/V 所有权通常按请求隔离。多个线程能看到同一个对象，不表示它可以被并发修改。

## 同步调用与异步工作

同步调用在返回前完成调用者需要的结果。异步调用先登记工作，完成结果稍后通过 future、callback、event 或队列出现。“同步/异步”描述完成通知方式，“串行/并行”描述工作是否同时执行，两组概念不能互相替代。

| 场景 | 调用返回时发生了什么 | 学习者要检查什么 |
| --- | --- | --- |
| 同步 CPU 函数 | 结果或错误已经可见 | 返回值、异常和对象状态 |
| CPU Worker 线程 | 主线程等待 `join()` 后收集结果 | 输入是否隔离、异常怎样回到请求 |
| CUDA 异步提交 | 工作可能仍在设备队列中 | stream 顺序、Event 和 buffer 生命周期 |
| SSE 写回 | 每次写入同步返回结果，整个请求持续多次写入 | 断连、背压、词元顺序和终态 |

异步不等于更快。它先改变控制流和资源生命周期；只有可重叠的工作足够多，并且同步点没有把重叠抵消时，才可能提高吞吐。

## 互斥、队列与单线程状态机

共享状态最容易从一个明确规则开始：只有一个线程修改状态，其他线程通过队列提交事件。另一种做法是在每次访问共享对象前加互斥锁。锁能阻止数据竞争，但如果整个模型执行期间都持锁，请求仍然串行。

```text
连接线程 --RequestEvent--> 单线程 ServiceCore --StepBatchPlan--> 模型执行
                ^                    |
                |-----完成事件--------|
```

当前 Chapter10 的 cpp-httplib Transport 使用互斥锁串行调用 `HttpAdapter` 和 `ServiceCore`。`RequestWorkerPool` 可以在线程中重叠独立 Context，Chapter08 Tensor Batch 则把多请求激活交给一次模型调用；三者解决的问题不同。

## 验证与练习

先完成 [`FQ07 Token Queue`](../exercises/foundation/problems/fq07_token_queue/README.md) 和 [`FQ11 Request Queue`](../exercises/foundation/problems/fq11_request_queue/README.md)，检查 FIFO、容量、取消和失败路径。随后运行：

```bash
bash code/chapter10_serving/scripts/run.sh
```

`chapter10_request_worker_test` 证明两个独立 Request/Executor 可以在线程中重叠；`chapter10_service_core_test` 证明多请求状态和 Tensor Batch 由 Scheduler 组织。完成后应能解释“多个连接”“多个线程”“多个 running 请求”和“一次 Tensor Batch”为什么不是同一件事。
