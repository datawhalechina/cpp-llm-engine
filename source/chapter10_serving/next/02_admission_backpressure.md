# 10.2 在执行前处理准入、容量和排队

准入回答“这个请求现在能否由服务接管”。<a href="https://github.com/datawhalechina/cpp-llm-engine/blob/main/code/chapter10_serving/src/service_core.cpp"><code>ServiceCore::admit()</code></a> 先检查 request ID、请求状态和实例容量，再把请求加入 waiting 队列；模型是否在当前 step 执行由 Chapter08 决定。

## 准入结果

| 结果 | 条件 | HTTP 常见映射 |
| --- | --- | --- |
| `Accepted` | 请求合法且容量足够 | 进入同步或流式处理 |
| `DuplicateId` | 活跃集合中已有同 ID | `409 Conflict` |
| `QueueFull` | waiting + running 达到上限 | `429 Too Many Requests` |
| `InvalidRequest` | 请求已终止或参数不合法 | `400 Bad Request` |

## 分清三种容量限制

| 限制 | 保护的资源 | 生效位置 |
| --- | --- | --- |
| `max_inflight` | 服务持有的请求与上下文总量 | admission |
| `max_running` | 同时进入调度候选的请求数 | waiting → running |
| token/block budget | 每个 step 的计算和 K/V 容量 | Chapter08 Scheduler |

waiting 队列提升请求时先比较基础优先级；同优先级先选择估算缓存前缀更长的请求，再用入队序号打破平局。估算阶段只读取索引，请求获得运行槽时才重新查询、采用真实前缀并增加块引用。Chapter08 的有界老化从请求进入 running 后开始累计。当前调度采用非抢占方式，运行中的请求持续执行到终态。

服务写回变慢时，有界队列负责限制内存增长。Writer 返回背压时，当前实现取消请求并释放资源。若要暂停后继续原请求，还需增加暂停状态、传输层可写通知和恢复入口。

## 验证容量和拒绝结果

运行 `bash code/chapter10_serving/scripts/run.sh`。`chapter10_service_core_test` 应固定重复 ID、队列容量、running 槽位和优先级提升结果；`chapter10_http_contract_test` 应将准入结果映射为稳定 HTTP 状态。`10.3` 继续验证已接收请求在各种终态下如何释放。
