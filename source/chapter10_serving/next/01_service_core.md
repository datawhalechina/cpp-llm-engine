# 10.1 服务核心（ServiceCore）只编排生命周期

`ServiceCore` 接收请求、保存服务状态并调用 Chapter08 调度器。<a href="../../_code/chapter10_serving/include/chapter10_serving/service_core.h"><code>service_core.h</code></a> 公开准入、单步执行、取消、超时和指标接口；它不计算 logits、不决定 K/V 物理布局，也不解析 socket 字节。

## 调用链与所有权

| 对象 | 主要所有者 | ServiceCore 怎样使用 |
| --- | --- | --- |
| `Request`、`Runtime`、`Executor` | Chapter06 | 保存引用并推进生成 |
| `KvCacheTransaction`、K/V payload | Chapter07 | 请求终态时触发释放 |
| `StepBatchPlan`、Scheduler | Chapter08 | 提交当前 running 集合和预算 |
| 性能证据 | Chapter09 | 输出固定 workload 所需指标 |
| HTTP 连接与 Writer | Chapter10 Transport | 接收事件，不进入模型对象 |

## 服务状态只保存一份

一个请求在 ServiceCore 中只有一个条目，依次处于 waiting、running 和 terminal。Request 内部保存生成阶段，Scheduler 读取它但不复制；网络 Adapter 通过 request ID 观察结果，不维护第二套模型状态。

| 服务动作 | 可以改变什么 | 不能改变什么 |
| --- | --- | --- |
| `admit` | waiting 队列、准入指标 | 模型状态和 K/V 内容 |
| `dispatch_step` | waiting/running、调度统计 | Scheduler 选择规则 |
| `cancel/timeout` | Request 终态 | 已发送的 HTTP headers |
| `release_terminal` | 服务条目和资源所有权 | 已归档的指标样本 |

依赖方向固定为 `Chapter10 ServiceCore → Chapter08 Scheduler → Chapter07 KV / Chapter06 Runtime`，服务代码不再调用 Chapter06 的旧调度实现。

`ServiceCore` 持有一个跨 step 复用的 `PriorityAgingPolicy`。admission 写入基础优先级和入队序号，空闲运行槽先提升高优先级 waiting 请求；进入 running 后，Chapter08 根据基础优先级和等待步数排序。配置批量执行器时，同一步的 Decode 候选通过一次模型 batch 调用推进，ServiceCore 不重新实现排序或模型计算。

## 验证生命周期编排

运行 `bash code/chapter10_serving/scripts/run.sh`。`chapter10_service_core_test` 应验证 waiting/running 转换、Chapter08 调度交接、批量执行和指标更新；通过后进入 `10.2`，观察请求在进入这条调用链前如何被接收或拒绝。
