# Chapter 10 代码：推理服务

本章把 Chapter06 请求状态、Chapter07 KV 资源和 Chapter08 调度器接入同一个服务生命周期，并通过与网络库无关的接口返回同步或流式结果。进入源码前先阅读 [`Chapter10 导读`](../../source/chapter10_serving/next/intro.md)。

## 代码架构

`CppHttplibServer` 只处理套接字和分块写出。真实服务入口把请求交给 `AsyncServiceLoop`，由后台单线程依次调用 `ServiceCore`；网络线程等待自己的词元事件并写回 Completion 或 SSE。`ServiceCore` 管理等待/运行状态、准入、指标和终态释放，再把每个执行步交给 Chapter08。模型状态由请求专属执行器与上下文持有，分页 K/V 通过 Chapter07 事务释放。

配置批执行器后，准入顺序是“绑定请求数据面 → 估算缓存前缀 → 进入等待队列”。估算不增加块引用，同优先级请求只用它决定提升顺序；获得运行槽时再重新查询并采用真实前缀，同步块表、`Request` 和模型上下文，实际长度随后传给 Chapter08 规划器。终态释放顺序是“发布完整缓存块 → 释放请求块引用 → 清理历史与服务条目”。`ServiceCore` 只调用生命周期钩子，不解析 Radix 树或修改 K/V 载荷。

绑定钩子同时支持连续缓存和分页缓存。DeepSeek CUDA 使用连续 MLA Cache 时，`bind_request()` 直接成功且不创建页表；工厂配置分页 MLA 后，同一钩子执行请求级块表绑定。`ServiceCore` 始终通过同一接口管理生命周期。

| 调用阶段 | 主要对象 | 所有权与失败处理 |
| --- | --- | --- |
| 网络接入 | `CppHttplibServer`、`ResponseWriter` | Transport 拥有连接；断连或写回失败转换为取消 |
| 协议映射 | `HttpAdapter` | Adapter 持有 Request，终态释放后才销毁 |
| 生命周期 | `ServiceCore` | 准入时绑定和估算，提升时采用缓存前缀，统一统计并释放资源 |
| 调度执行 | Chapter08 Scheduler、Executor | 每步提交候选；失败写入 Request 终态 |
| 缓存发布与回收 | Chapter07 cache transaction、Engine Factory | 终态先发布可复用块，再释放请求引用 |

异步入口允许多个 HTTP Completion/SSE 先进入等待队列，再由同一个 Chapter08 调度步选择。`AsyncRequestHandle` 分别保存词元事件和终态，网络线程不直接修改 `ServiceCore`。`AsyncServiceLoop` 的可选提交收集窗口会在服务空闲时短暂等待同批请求；默认值为零，模型入口可用数毫秒延迟换取更稳定的组批机会。`RequestWorkerPool` 验证独立上下文的 CPU 线程重叠；Chapter08 张量组批验证一次模型调用处理多请求。异步接入、工作线程和模型数据面是三个不同层次。

## 阅读顺序与资产

| 模块 | 学习内容 | 验证 |
| --- | --- | --- |
| `ServiceCore` | 准入、缓存采用、等待/运行队列、调度交接、指标和统一释放 | `chapter10_service_core_test` |
| `AsyncServiceLoop` | 将网络提交、取消和超时投递给单线程服务状态机 | `chapter10_async_service_test` |
| `CompletionRequestCollector` | 在短窗口内收集请求并调用一次模型专属准备 batch | `chapter10_request_preparer_test` |
| `RequestWorkerPool` | 用独立 Request 与 Executor 并行执行 CPU 请求 | `chapter10_request_worker_test` |
| `ResponseWriter` | 表达响应头、SSE 事件、断连和背压 | `chapter10_http_contract_test` |
| `HttpAdapter` | 把 health、metrics 和 Completion 路由映射到 ServiceCore | `chapter10_http_contract_test` |
| `CppHttplibServer` | 把可选 cpp-httplib socket 接到 HttpAdapter | `run_cpp_httplib_loopback.sh` |
| 终态释放 | 从等待/运行集合移除请求并释放 K/V 与批执行状态 | `chapter10_terminal_release_test` |

1. 阅读 `ServiceCore::admit()` 和 `promote_waiting()`，理解服务容量与调度预算的区别。
2. 阅读 `dispatch_step()`，观察服务如何把运行请求交给 Chapter08，而不是重新实现调度。
3. 阅读 `release_terminal()`，确认完成、取消、超时和失败共用一条资源释放路径。
4. 阅读 `HttpAdapter` 与 `ResponseWriter`，跟踪同步 Completion 和 SSE 如何处理断连、背压与异常。
5. 最后阅读 `CppHttplibServer`，查看网络库如何只承担 Transport 适配。

`HttpAdapter` 同时接受可选的文本分词回调。带 `prompt_tokens` 的测试直接构造稳定输入；带 `prompt` 的真实服务在应用组合层注入 Tokenizer。Adapter 只调用回调并接收词元序列，不依赖具体词表实现。

请求从 waiting 进入 running 后，由 Chapter08 每步选择执行对象。请求到达终态并不等于资源已经释放；调用 `release_terminal()` 后，Request 缓存、批执行绑定和分页 KV 块才会一起回收。

## 运行与验证

默认脚本使用进程内适配器，运行服务状态、终态释放、HTTP/SSE、异步循环、请求准备收集器和并行工作线程测试：

```bash
bash code/chapter10_serving/scripts/run.sh
```

`chapter10_terminal_release_test` 在 `executor.prefill` 注入模型失败，`chapter10_http_contract_test` 在 `transport.write` 注入 SSE 写回异常；两者都检查请求终态、指标、inflight 数量和缓存释放。修改失败映射或所有权代码后运行：

```bash
npm run test:fault-injection
```

可选 socket 闭环使用：

```bash
bash code/chapter10_serving/scripts/run_cpp_httplib_loopback.sh
```

准备本地 `httplib.h` 后可以设置 `CHAPTER10_CPPHTTPLIB_INCLUDE_DIR`；未设置时，CMake 获取固定版本 `v0.18.3`。该入口验证 health、metrics、异步 Completion 和 SSE；模型是否形成 batch 还取决于同时到达的请求与 Batch Runner。

TinyLlama 使用真实 socket 回环验证默认 CPU 服务，Qwen3 使用独立固定模型入口验证第二种稠密架构、双请求调度和 Radix 命中：

```bash
bash code/engine/scripts/run_real_http_server.sh --check
bash code/chapter10_serving/scripts/run_qwen3_serving.sh
```

两种稠密模型的 CUDA 服务保持 `ServiceCore` 与协议层不变，只替换 Executor：

```bash
bash code/engine/scripts/run_real_cuda_serving.sh
bash code/engine/scripts/run_qwen3_cuda_serving.sh
```

本地已有 Qwen1.5-MoE 时，可以验证两个运行请求 Decode 期间再加入两个请求的动态 admission：

```bash
bash code/chapter09_performance/scripts/run_qwen15_moe_service_ab.sh
```

EXP-CPU-18 检查完整生成序列、服务生命周期和分组 MoE 性能；Transport 不在计时范围内。

本地已有 DeepSeek-V2-Lite 时，可以验证动态请求接纳、协议映射与终态释放：

```bash
bash code/chapter10_serving/scripts/run_deepseek_v2_lite_service.sh
```

有 NVIDIA GPU 时，可以把 DeepSeek CUDA Executor 接入同一 `ServiceCore`，验证真实生成、双请求专家 batch、流式回调、取消和终态释放：

```bash
CUDA_HOME=/usr/local/cuda-12.8 \
bash code/engine/scripts/run_deepseek_v2_lite_cuda_serving.sh
```

该入口使用固定词元请求，不经过 Tokenizer 或 HTTP Adapter。两个活跃请求先各自完成 Prefill，再由 Chapter08 把 Decode 交给一次 CUDA batch runner；测试同时检查持久化工作区、26 层专家批执行和取消请求释放。

需要继续观察真实 HTTP 字节与 SSE chunk 时运行：

```bash
bash code/engine/scripts/run_deepseek_v2_lite_http_server.sh --check
```

该命令复用现有 cpp-httplib Transport，完成本地 health、Completion、SSE 和 metrics 回环；去掉 `--check` 可以保留服务供手工请求。

CUDA 文本服务使用独立入口，避免把 CPU 与设备执行结果混在同一个脚本中：

```bash
CUDA_HOME=/usr/local/cuda-12.8 \
bash code/engine/scripts/run_deepseek_v2_lite_cuda_http_server.sh --check
```

该回环以文本 `prompt` 验证真实分词器、CUDA 工厂、分页 MLA、异步 Completion/SSE 和终态释放。并发连接可共同进入等待/运行集合；是否形成张量 batch 仍取决于 Chapter08 计划和 CUDA Batch Runner。

SmolVLM 通过 Chapter10 的请求准备收集器，把两个 JSON/Base64 图片请求交给一次 CUDA Vision batch，再分别建立连续嵌入和请求 Context。下面的入口验证图文生成、Completion/SSE、视觉组批、语言 Prefill/Decode 组批、请求隔离和终态释放；HTTP Adapter 与 ServiceCore 仍保持模型无关。

```bash
CUDA_HOME=/usr/local/cuda-12.8 \
bash code/multimodal/scripts/run_cuda_smolvlm_context.sh \
  <mmproj.gguf> <model.gguf> <image.ppm> mixed-service
```

## 扩展入口

`ServiceCore` 已接通 Chapter08 优先级策略、异步请求事件循环、真实解码张量组批和分页 K/V 数据。当前背压会取消并释放请求；后续可以增加自动截止时间和异步写出器暂停/恢复。真实多请求预填充张量组批仍属于 Chapter08 数据面能力，需要由调度计划和批执行器共同实现。
