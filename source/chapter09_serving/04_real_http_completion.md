# 09.4 提供真实模型 HTTP Completion

应用启动时加载一次 `RealLlamaExecutorFactory`。每个 completion 通过 Factory 创建独立 Executor/Context，再与 `Request` 一起交给 `ServiceCore`；同步 JSON 和 SSE 只改变输出方式，不改变模型执行与资源所有权。

## 从 HTTP 请求进入真实模型

当前最小请求接收 `prompt_tokens`。真实 tokenizer 位于 Engine 应用组合层，负责把文本转换为 token ID；Transport、ServiceCore 和模型 Context 不重复实现分词逻辑。

| 阶段 | 对象 | 可验证结果 |
| --- | --- | --- |
| 应用启动 | `RealTokenizerBackend`、Factory | 词表一致，Model 只加载一次 |
| 请求创建 | `RequestExecutorFactory` | 每个请求获得独立 Executor/Context |
| 同步 completion | `HttpAdapter::handle()` | 固定 token ID 与 finish reason |
| SSE completion | `handle_stream()`、Writer | token 顺序固定，最终事件完整 |
| 终态回收 | `release_completed()` | cache、Executor 和 Context 释放 |
| 指标读取 | `GET /metrics` | accepted、completed、released 与 token 数一致 |

## 运行本地 HTTP 闭环

准备 TinyLlama GGUF 后，下面的入口构建 server、临时监听本地端口、执行健康检查、同步 completion、SSE 和 metrics，然后退出：

```bash
CPP_LLM_ENGINE_MODEL_PATH=/path/to/tinyllama.gguf \
bash code/engine/scripts/run_real_http_server.sh --check
```

默认监听 `127.0.0.1:18082`。去掉 `--check` 可以保留服务供手工请求；`REAL_HTTP_BIND_ADDRESS` 和 `REAL_HTTP_PORT` 可修改地址。当前 cpp-httplib transport 用互斥锁串行调用 ServiceCore，请求级并发 API 尚未接入 SSE。

| 请求 | 固定结果 |
| --- | --- |
| `Hello TinyLlama` 同步 JSON | `29892 306 29915 29885` |
| `Hi` SSE | `839 744 29892 263` |
| 两请求累计 metrics | accepted/completed/released=`2`，generated tokens=`8` |
| 断开或超时 | 请求进入对应终态并释放资源 |

## 验证与衔接

读完后应能沿 `HTTP -> HttpAdapter -> Factory -> ServiceCore -> Runtime -> Executor` 追踪一次请求，并说明 tokenizer、transport 与模型 Context 分别由谁持有。文本 prompt API、鉴权和异步 transport 应继续复用该对象链，而不是在 Gateway 中复制模型执行逻辑。
