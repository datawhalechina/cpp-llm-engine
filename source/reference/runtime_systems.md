# 推理引擎运行时

Generation Runtime 持续推进 Request，并协调 Scheduler、KV Cache、输出和终态资源释放。

## 1. Runtime 管理的状态

单次 `forward` 产生一组数值结果；Runtime 还要持续接收和调度请求、控制资源、处理异常并输出生成事件。

## 2. 运行时的核心边界

最小 generation runtime 已经给出了基本骨架：

- `prefill` 和 `decode` 分开
- `Request` 状态显式
- `KV Cache` 显式
- `Scheduler` 显式
- `Sampler` 显式

真实引擎在这个基础上，还要把下面这些边界继续收紧：

- 请求生命周期
- 并发请求管理
- continuous batching
- backpressure
- 请求取消
- 超时
- 流式输出
- 优先级队列

这些不是“额外功能”，而是运行时必须有的控制面。

它们的最小落点可以按三层理解：服务层负责接入、限流和取消；scheduler 负责 admission、优先级和每 step 的 token budget；runtime 负责执行、输出和终态通知。KV Cache、workspace 等资源回收必须由终态路径统一触发，不能由流式回调或 API 层各自释放。

## 3. 请求生命周期

请求生命周期至少应该能回答这些问题：

- 一个请求什么时候进入系统
- 它什么时候进入 `prefill`
- 它什么时候切到 `decode`
- 它什么时候结束
- 它取消时资源怎么释放

如果这些边界不清楚，后面出现内存泄漏、cache 泄漏或调度卡死时，很难知道问题属于哪一层。

## 4. 并发请求和 continuous batching

生产 Serving 通常需要同时管理多个请求。

最小 runtime 可以先只服务单请求，但系统设计上必须预留这些能力：

- 多请求同时在队列里存在
- 不同请求的 `prefill` 和 `decode` 可以交错推进
- scheduler 能够把多个请求合成稳定的 batch
- 新请求进入后不会打乱已有请求的状态

continuous batching 的重点不是“批处理更大”，而是“请求不断进入时，系统还能持续推进”。

## 5. Backpressure 和取消

一旦系统开始持续接请求，就必须能处理压力变化。

backpressure 解决的是：

- 请求太多时，系统怎么慢下来
- 队列太长时，怎么避免把所有资源都吃满
- 下游变慢时，上游怎么得到反馈

取消解决的是：

- 用户不需要了，系统怎么停
- 资源已经占住时，怎么回收
- 取消之后，哪些状态还能复用，哪些必须丢弃

这两项都属于 runtime 的基本行为，不是附加优化。

## 6. 流式输出和优先级

流式输出让用户能看到生成过程，不只是最后结果。  
优先级队列让系统能区分不同请求的重要程度，而不是永远按到达顺序处理。

这两项一旦进入系统，scheduler 就不只是“选下一个请求”，而是在约束多个目标：

- 吞吐
- 延迟
- 公平性
- 可观察性

教程实现采用 FIFO、显式状态和最少 prefill 进度优先。生产系统如果引入业务优先级，还必须同时定义公平性、饥饿保护、取消后的队列清理和可验证的 token budget，否则优先级只会改变顺序，不能证明服务质量改善。

## 7. 和章节主线的关系

| 章节 | 教程实现 | 生产系统常见扩展 |
| --- | --- | --- |
| `chapter06_generation` | generation runtime、chunked prefill、基础公平性、token/block-budget batch | 抢占、优先级和并行 batch executor |
| `chapter05_quantization` | 稳定 runtime 上的表示变化 | 资源边界和调度边界继续保持清楚 |
| `chapter07_benchmark` | 验证和结果记录 | 运行时缺口对应的性能和稳定性问题 |
| `chapter09_serving` | waiting/running 队列、`dispatch_step()`、Block KV ownership、终态释放 | device K/V payload、抢占和业务优先级 |

运行时系统参考页在主线 runtime 骨架上继续补充服务化能力。

## 8. 使用顺序

- 先完成 `chapter06_generation` 的主线阅读
- 再回到这里确认真实 serving runtime 还缺什么
- 再根据系统缺口决定后续先补调度、内存还是服务化

如果只想抓最关键的结论，记住一句话就够了：  
`chapter06` 解决“能持续生成”，运行时系统参考页补充“能持续服务”。
