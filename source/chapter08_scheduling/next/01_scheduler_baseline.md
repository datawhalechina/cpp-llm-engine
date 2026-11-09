# 08.1 从调度规则和固定轮转开始

Generation Core 能推进单个请求，Chapter07 能为请求提供可提交的 K/V 位置。Scheduler 决定当前执行机会交给谁，并把选择结果交给 Runtime；它不计算 logits，也不保存 K/V 数值。

## 调度器读取和产生什么

`ScheduledRequest` 保存请求、采样器、Executor 和调度元数据的引用。请求状态仍由 Chapter06 持有，ServiceCore 负责请求进入和退出运行集合。

| 对象 | 用途 | 状态所有者 |
| --- | --- | --- |
| `request_id` | 连接请求、输出事件和缓存页表 | Chapter10 ServiceCore |
| `Request` | 阶段、Prefill 进度和生成结果 | Chapter06 |
| `Sampler`、`Executor` | 采样与请求专属模型 Context | Chapter06 模型执行路径 |
| `SchedulingMetadata` | 基础优先级和入队序号 | Chapter08 策略调用方 |
| `StepBatchPlan` | 本步选择谁、处理多少词元 | Chapter08 Planner，只读产生 |
| `StepBatchStats` | 实际推进、延后和生成数量 | Chapter08 执行阶段 |

## Round-robin 提供比较基线

固定轮转按输入顺序推进可运行请求。它没有词元预算、分块 Prefill 和缓存容量判断，但顺序稳定，适合先验证多请求隔离和流式事件归属。

```python
while any_progress:
    any_progress = False
    for item in scheduled_requests:
        if item.request is invalid or terminal:
            continue
        before = item.request.generated_size
        runtime.decode_one(item.request, item.sampler, item.executor)
        if item.request.generated_size > before:
            emit(item.request_id, item.request.newest_token)
            any_progress = True
```

| 场景 | 固定轮转动作 | 可观察结果 |
| --- | --- | --- |
| 请求可运行 | 推进一步 | 最多产生一个新词元 |
| 请求已终止 | 跳过 | Context 和输出不再变化 |
| Executor 失败 | Request 进入失败终态 | 不产生占位词元 |
| 一整轮无进展 | 退出 | 避免空转 |

## 验证基线

```bash
cmake -S code/chapter08_scheduling -B /tmp/cpp-llm-chapter08-baseline \
  -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/cpp-llm-chapter08-baseline \
  --target chapter08_scheduler_test -j2
ctest --test-dir /tmp/cpp-llm-chapter08-baseline \
  -R '^chapter08_scheduler_test$' --output-on-failure
```

| 通过条件 | 对应意义 |
| --- | --- |
| 终态请求不再执行 | 生命周期状态唯一 |
| Executor 失败不采样 | 错误不会伪造输出 |
| 回调携带正确请求 ID 和步号 | 多请求输出不串线 |
| 顺序与批量教学结果一致 | 后续优化不改变逐请求语义 |
