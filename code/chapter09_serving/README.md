# Chapter 09：Serving Extensions 服务化扩展

本目录把 Chapter 06 的 `Request`、`Runtime`、`Scheduler`、`RequestModelExecutor`、`KVCache` 和 `KvBlockPool` 接到 admission、等待/运行队列、HTTP/SSE 和完成后资源回收。完整学习说明见 [`source/chapter09_serving`](../../source/chapter09_serving/intro.md)。

## 资产总览

| 资产 | 作用 |
| --- | --- |
| `include/chapter09_serving/serving_core.h` | Service Core 状态、admission 和统计契约 |
| `src/serving_core.cpp` | 队列推进、请求级 CPU 并发、token/block budget、取消/超时、资源回收和在线延迟指标 |
| `include/chapter09_serving/http_adapter.h` | transport-neutral HTTP request/response 契约 |
| `src/http_adapter.cpp` | `/health`、`/metrics`、同步/SSE completion 和 Executor Factory 映射 |
| `include/chapter09_serving/response_writer.h` | 不绑定网络库的 headers、event、断开和背压契约 |
| `include/chapter09_serving/cpp_httplib_transport.h` | 可选 cpp-httplib server 适配器 |
| `src/cpp_httplib_transport.cpp` | HTTP endpoint、chunked SSE 和 ResponseWriter 适配 |
| `examples/main_cpp_httplib.cpp` | 可选真实 HTTP server 启动入口 |
| `examples/main_serving.cpp` | 两轮 batch dispatch 演示 |
| `tests/chapter09_serving_test.cpp` | 准入、同步执行、失败和终态释放测试 |
| `tests/step_dispatch_test.cpp` | token/block budget、动态补位和块回收测试 |
| `tests/parallel_dispatch_test.cpp` | 独立 Executor 请求级并行与结果隔离测试 |
| `tests/cancellation_release_test.cpp` | 部分 Prefill 取消后的 cache/block 回收测试 |
| `tests/service_metrics_test.cpp` | 手动时钟下 queue wait、TTFT 和 completion 测试 |
| `tests/http_adapter_test.cpp` | endpoint、状态码、completion 和 metrics 测试 |
| `tests/response_writer_test.cpp` | 正常写出、客户端断开、背压和 finish 测试 |
| `tests/http_adapter_writer_test.cpp` | HttpAdapter 通过 writer 输出 token event 和终态 |
| `../engine/examples/real_tinyllama_serving_smoke.cpp` | 可选真实 Model、双请求 Context 隔离和释放验证 |
| `../engine/examples/real_tinyllama_http_adapter_smoke.cpp` | 可选真实同步/SSE completion 与 metrics 验证 |
| `../engine/examples/real_tinyllama_serving_benchmark.cpp` | 可选真实 prefill/decode、TTFT、完成时间和吞吐测量 |
| `results/tinyllama_service_cpu_baseline.json` | 最近一次可复查的真实 CPU 服务基线 |
| `results/tinyllama_service_*_avx2.json` | 相同真实 workload 的 sequential/parallel-request A/B |
| `scripts/run.sh` | 隔离构建、demo 和 CTest 入口 |
| `scripts/run_cpp_httplib_loopback.sh` | 构建可选 transport 并验证真实 HTTP/SSE loopback |
| `../engine/scripts/run_real_http_server.sh` | 组装真实 TinyLlama Model/Context 并启动或检查本地 HTTP 服务 |

## 运行与验证

从仓库根目录执行：

```bash
bash code/chapter09_serving/scripts/run.sh
```

完成标准：请求容量达到上限时返回 `QueueFull`；`dispatch_step()` 检查 token budget 和可选 block pool；`dispatch_parallel_requests()` 让独立 Executor/Context 在多个 CPU worker 上重叠；终态资源均可释放。并发路径不支持 BlockTable 或 worker 内 streaming，也不代表 Tensor batch。

默认脚本使用 fake executor，保证 Chapter09 不依赖模型文件。真实 TinyLlama 验证由 `code/engine` 的可选构建承接：同一个 Factory 只加载一份 Model，再为每个 ServiceCore request 创建独立 Context。构建命令见 `code/engine/README.md`。

真实服务 benchmark 运行 `bash code/engine/scripts/run_real_serving_benchmark.sh`。脚本默认执行 `warmup=1`、`iterations=3` 和 scalar policy；设置 `REAL_SERVING_EXECUTION_POLICY=optimized-avx2` 可显式组合已验证的 Attention/FFN AVX2 backend。性能数字只用于相同环境的 A/B 比较；golden token、Context/cache 隔离和 terminal 释放仍是硬性通过条件。

请求级并发 A/B 运行 `bash code/engine/scripts/run_real_parallel_requests_ab.sh`。它固定相同 optimized AVX2 backend，只切换 `sequential` 与 `parallel-requests` dispatch mode，并分别归档墙钟、吞吐和重叠计算时间。

HTTP adapter 只定义映射，不创建 socket。当前支持 `GET /health`、`GET /metrics`、同步 `POST /v1/completions` 和 SSE response；Engine 可注入 `RealLlamaExecutorFactory`，同步响应会返回 `generated_token_ids`。客户端断开返回 `499`，deadline 返回 `408`，两条路径都会触发 ServiceCore 的终态和资源回收。

cpp-httplib transport 默认不构建。需要真实 HTTP 时，使用 `-DCHAPTER09_ENABLE_CPPHTTPLIB=ON`，CMake 会获取固定版本 `v0.18.3`；未开启时，Chapter09 不需要网络访问即可完成全部核心测试。

如果已经准备好 `httplib.h`，可以避免 CMake 在线拉取：

```bash
CHAPTER09_CPPHTTPLIB_INCLUDE_DIR=/path/to/cpp-httplib \
bash code/chapter09_serving/scripts/run_cpp_httplib_loopback.sh
```

真实 TinyLlama CPU 服务从仓库根目录启动：

```bash
CPP_LLM_ENGINE_MODEL_PATH=/path/to/model.gguf \
bash code/engine/scripts/run_real_http_server.sh
```

另开终端向 `http://127.0.0.1:18082` 请求 `/health`、`/metrics` 或 `/v1/completions`。也可以执行同一脚本的 `--check` 模式验证同步 golden、SSE token 顺序和 metrics。当前 HttpAdapter 仍使用串行 ServiceCore 路径；请求级并发 API 尚未接入 SSE transport。
