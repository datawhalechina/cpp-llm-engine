# 10.1 服务核心（ServiceCore）只编排生命周期

`ServiceCore` 接收请求、保存服务状态并调用 Chapter08 调度器。<a href="https://github.com/datawhalechina/cpp-llm-engine/blob/main/code/chapter10_serving/include/chapter10_serving/service_core.h"><code>service_core.h</code></a> 公开准入、单步执行、取消、超时和指标接口。logits 由执行器计算，K/V 物理布局由 Chapter07 管理，套接字字节由传输层解析。

## 调用链与所有权

| 对象 | 主要所有者 | ServiceCore 怎样使用 |
| --- | --- | --- |
| `Request`、`Runtime`、`Executor` | Chapter06 | 保存引用并推进生成 |
| `KvCacheTransaction`、K/V 载荷 | Chapter07 | 请求终态时触发释放 |
| `StepBatchPlan`、Scheduler | Chapter08 | 提交当前 running 集合和预算 |
| 性能证据 | Chapter09 | 输出固定负载所需指标 |
| HTTP 连接与 Writer | Chapter10 Transport | 接收事件，不进入模型对象 |

## 服务状态只保存一份

一个请求在 `ServiceCore` 中只有一个条目，依次处于等待、运行和终态。`Request` 内部保存生成阶段，调度器读取它但不复制；网络适配器通过请求 ID 观察结果，不维护第二套模型状态。

| 服务动作 | 更新的状态 | 交给其他层的工作 |
| --- | --- | --- |
| `admit` | 执行器绑定、缓存前缀估算、waiting 队列和准入指标 | 块引用、模型位置与 K/V 数值计算由缓存和执行器处理 |
| `dispatch_step` | 等待/运行集合、调度统计 | 候选选择规则由调度器处理 |
| `cancel/timeout` | Request 终态 | 已发送的 HTTP 响应头保持原状 |
| `release_terminal` | 服务条目和资源所有权 | 已记录的指标样本继续保留 |

依赖方向固定为 `Chapter10 ServiceCore → Chapter08 Scheduler → Chapter07 KV / Chapter06 Runtime`，服务代码不再调用 Chapter06 的旧调度实现。

准入先让 `ExecutorBatchRunner` 绑定请求级数据面，再估算可复用前缀。估算不修改 Radix 统计、块引用、Request 或模型 Context。请求获得空闲运行槽时，CPU/CUDA Factory 重新查询并采用仍然有效的前缀，同步共享块表、Request Prefill 游标和模型 Context；采用失败会进入失败终态并释放已绑定资源。

“绑定”是可选资源钩子。连续 K/V 或连续 MLA 上下文直接返回成功；工厂配置共享块池后，钩子建立请求 ID、页表和物理载荷之间的关系。

`ServiceCore` 持有一个跨模型步复用的 `PriorityAgingPolicy`。准入时记录基础优先级、估算前缀长度和入队序号。空闲运行槽先选高优先级请求；同级时选估算命中更长者，再用入队序号破平局。进入运行状态后，Chapter08 使用实际采用长度继续排序。配置批执行器时，同一步的解码候选通过一次模型批量调用推进。

## 验证生命周期编排

运行 `bash code/chapter10_serving/scripts/run.sh`。`chapter10_service_core_test` 验证估算无状态提交、提升时采用、过期估算失败释放、等待/运行转换、Chapter08 调度、批量执行和指标更新。有 TinyLlama 时，`real_tinyllama_serving_test` 进一步验证等待请求未固定缓存块、Radix 命中请求先执行，且终态只释放请求引用，不把索引保留块误判为泄漏。
