# 10.4 让传输层只处理连接和字节

传输层（Transport）负责监听、HTTP 解析、响应写出和连接事件；<a href="https://github.com/datawhalechina/cpp-llm-engine/blob/main/code/chapter10_serving/include/chapter10_serving/http_adapter.h"><code>HttpAdapter</code></a> 负责路由与参数语义；`ServiceCore` 负责请求生命周期。更换网络库时，后两层接口约定保持不变。

## 三层映射

| 层 | 输入与输出 | 交给下一层的内容 |
| --- | --- | --- |
| 传输层 | 套接字、HTTP 方法/路径/请求体、分块数据 | 将解析后的请求和连接事件交给适配器 |
| `HttpAdapter` | Completion 参数、状态码、SSE 事件 | 将类型化请求和写回结果交给服务核心 |
| `ServiceCore` | 准入、执行、取消、指标 | 将模型步交给调度器和执行器 |

## 网络库选择

| 方案 | 教程中的位置 | 主要取舍 |
| --- | --- | --- |
| cpp-httplib | 首个真实 HTTP/SSE 回环 | 接入短，适合验证同步 HTTP/1.1 |
| Boost.Asio/Beast | 异步连接、Deadline 和背压扩展 | 生命周期完整，状态机更复杂 |
| Crow/oat++ | Web 框架对照 | 路由功能丰富，需要额外标出服务核心边界 |

`ResponseWriter` 是传输层与适配器之间的窄接口，只暴露响应头、事件、结束和连接状态。网络库对象停留在这层接口之外，`ServiceCore` 只接收与请求生命周期有关的结果。

服务进程的模型路径、监听地址、端口、后端和实现策略由命令行或部署环境提供。传输层在启动时完成解析，再把类型化配置交给执行器工厂和 `ServiceCore`；请求处理期间直接使用已解析配置。配置优先级是：显式命令行参数、环境默认值、程序内安全默认值，启动日志打印最终生效值。请求自身的生成参数仍由 `HttpAdapter` 校验，与进程级后端配置分开。

cpp-httplib 可以并发接收连接，`ServiceCore` 仍按单线程状态机运行。真实服务入口通过 `AsyncServiceLoop` 连接两者：网络线程提交类型化请求并等待自己的词元事件，后台线程是 `ServiceCore` 的唯一调用者。多个连接可以先进入等待队列，再共同参与 `StepBatchPlan`；服务状态仍保持串行修改。

## 验证传输层可替换性

先运行 `bash code/chapter10_serving/scripts/run.sh`，确认 `chapter10_http_contract_test` 的同步兼容接口和 `chapter10_async_service_test` 的异步事件交付。本地已有 cpp-httplib 时，再运行 `bash code/chapter10_serving/scripts/run_cpp_httplib_loopback.sh`，通过条件是 health、metrics、Completion 和 SSE 回环全部成功。`10.5` 继续追踪一次 Completion。
