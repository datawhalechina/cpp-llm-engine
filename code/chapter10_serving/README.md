# Chapter 10 代码资产：推理服务

本目录把 Chapter06 Request/Runtime、Chapter07 KV 资源和 Chapter08 Scheduler 接入同一个服务生命周期。学习说明见 [`Chapter10 导读`](../../source/chapter10_serving/next/intro.md)。先运行：

```bash
bash code/chapter10_serving/scripts/run.sh
```

## 资产总览

| 资产 | 职责 | 验证 |
| --- | --- | --- |
| `service_core.h/.cpp` | 准入、waiting/running、step dispatch、终态与指标 | `chapter10_service_core_test` |
| `request_worker.h/.cpp` | 请求级 CPU Worker 与独立 Executor/Context | `chapter10_request_worker_test` |
| `response_writer.h` | headers、SSE event、断连和背压接口 | `chapter10_http_contract_test` |
| `http_adapter.h/.cpp` | health、metrics、Completion 和 SSE 映射 | `chapter10_http_contract_test` |
| `cpp_httplib_transport.h/.cpp` | 可选真实 HTTP/SSE socket 适配 | `run_cpp_httplib_loopback.sh` |
| `terminal_release_test.cpp` | waiting/running 与 K/V 的统一释放 | `chapter10_terminal_release_test` |

默认构建不访问网络。准备本地 `httplib.h` 后，可设置 `CHAPTER10_CPPHTTPLIB_INCLUDE_DIR` 运行 `scripts/run_cpp_httplib_loopback.sh`；未设置时，CMake 获取固定版本 `v0.18.3`。ServiceCore 已接通 Chapter08 优先级策略、真实 Tensor batch 和分页 K/V payload；Engine 负责 TinyLlama HTTP 与 sequential/parallel Worker 验证。当前背压会取消并释放请求，自动 Deadline 和异步 Writer 暂停/恢复仍待实现。
