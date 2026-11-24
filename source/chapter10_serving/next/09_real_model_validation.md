# 10.9 用真实模型验证服务

确定性执行器适合覆盖状态机。TinyLlama 与 Qwen3 验证稠密模型的 CPU/CUDA 服务，Qwen1.5-MoE 验证动态准入和专家分组能否穿过 `ServiceCore`，DeepSeek-V2-Lite 验证 MLA-MoE、Completion 与 SSE。模型复用同一组服务核心和调度器；进入 HTTP 的路径再复用适配器。工厂对象负责把具体模型装配到统一执行接口。

## 真实对象怎样组装

| 生命周期 | 对象 | 验证事实 |
| --- | --- | --- |
| 进程启动 | Tokenizer、共享 Model、Executor Factory | 权重只加载一次 |
| 请求创建 | 独立 Executor/Context | token history 和 K/V 不串请求 |
| 服务执行 | ServiceCore → Chapter08 → batch runner | golden token、批量调用与阶段一致 |
| HTTP 写回 | JSON 或 SSE | token 顺序和 finish reason 正确 |
| 请求终态 | release | Context、K/V 和服务条目归零 |

## 分层运行真实验证

| 入口 | 覆盖范围 | 还需补充的验证 |
| --- | --- | --- |
| 真实服务核心冒烟测试 | 模型/上下文、双请求组批和生命周期 | 网络端到端性能另测 |
| HTTP `--check` | 串行适配器入口下的套接字、Completion、SSE 和指标 | 并发 HTTP 与高并发压测另测 |
| CUDA 服务冒烟测试 | CUDA 执行器、分页解码组批、Completion/SSE、取消、超时和释放 | 融合 PagedAttention 与套接字服务分别验证 |
| 服务性能基准 | 固定负载的 TTFT、吞吐和内存 | 开放环生产负载另建实验 |
| 顺序/并行 A/B | 请求级 CPU 重叠 | 张量组批收益由批执行实验验证 |

归档结果使用 Chapter09 的负载 ID、环境和正确性字段。正文解释验证方法，具体机器上的数字进入实验资产索引。

双请求 ServiceCore smoke 在 admission 时绑定请求 ID 和分页数据面，再把 Decode 候选交给 `RealLlamaExecutorFactory::decode_batch()`。验证同时检查 batch 调用计数、逐请求 golden、`paged_payload_active`、Context 隔离、BlockPool 提交数和终态释放。请求级并行 Worker 仍使用独立连续 Context；它用于比较 CPU 请求重叠，不等同于共享模型的 Tensor batch。

## 验证真实服务行为

确认 TinyLlama GGUF 已存在后运行默认 socket 服务测试：

```bash
bash code/engine/scripts/run_real_http_server.sh --check
```

Qwen3 使用同一服务对象，但固定模型、prompt 和 golden 不同：

```bash
bash code/chapter10_serving/scripts/run_qwen3_serving.sh
```

两条 CPU 路径都要求同步 Completion、SSE、metrics 和固定 token golden 成立，两个请求的 Context/K/V 不串扰，Radix 前缀在调度提升时命中，终态后服务条目和请求块释放。`10.10` 将这些单进程行为映射到成熟服务架构。

Qwen1.5-MoE 使用 Chapter09 的动态准入实验验证 ServiceCore 生命周期：两个请求运行后再接纳两个请求，两种专家执行路径必须得到相同输出、批量调用和终态计数。

```bash
bash code/chapter09_performance/scripts/run_qwen15_moe_service_ab.sh
```

该入口不经过 `HttpAdapter` 或 socket，因此只证明 MoE Factory、Chapter08 批执行和 Chapter10 生命周期能够协作，不作为 Qwen1.5-MoE HTTP 服务证据。

DeepSeek-V2-Lite 先验证动态准入和双请求分组 MoE，再让同步 Completion 与 SSE 顺序经过同一个 Adapter：

```bash
bash code/chapter10_serving/scripts/run_deepseek_v2_lite_service.sh
```

该入口证明 DeepSeek Factory 能接入 Chapter10 的协议与生命周期接口。Adapter 测试不启动 socket，也不把两个顺序请求解释为并发 HTTP 组批。

需要观察真实 socket 与 SSE chunk 时，再执行：

```bash
bash code/engine/scripts/run_deepseek_v2_lite_http_server.sh --check
```

这个回环继续使用相同 Adapter 和 Factory；Transport 只负责连接与字节写回。

有 NVIDIA GPU 时再运行：

```bash
bash code/engine/scripts/run_real_cuda_serving.sh
```

Qwen3 使用相同 CUDA 服务入口，只替换模型与 golden：

```bash
bash code/engine/scripts/run_qwen3_cuda_serving.sh
```

两个入口都保持 `ServiceCore` 和 `HttpAdapter` 不变，只把每请求 Executor 换成 CUDA Context。TinyLlama 与 Qwen3 均已通过正常 Completion、SSE、客户端取消和超时验收，四个请求释放后 `inflight`、`waiting`、`running` 全部归零。

DeepSeek-V2-Lite 的 CUDA 路径先验证服务核心：

```bash
CUDA_HOME=/usr/local/cuda-12.8 \
bash code/engine/scripts/run_deepseek_v2_lite_cuda_serving.sh
```

该入口把 `RealDeepSeek2CudaExecutorFactory` 作为 batch runner 注入 `ServiceCore`。两个活跃请求分别完成 Prefill 后，共同进入一次 CUDA Decode batch；每个请求保留自己的压缩 MLA 历史，26 个 MoE 层复用同一个批量工作区执行专家 Gate、Up 和 Down。第三个请求在执行前取消。测试检查首个词元 `207`、两请求生成一致、流式顺序、批量调用计数、指标和三个终态的 Executor 释放。

最后运行文本到网络响应的 CUDA 组合验收：

```bash
CUDA_HOME=/usr/local/cuda-12.8 \
bash code/engine/scripts/run_deepseek_v2_lite_cuda_http_server.sh --check
```

进程启动时加载一次真实 Tokenizer、共享 CUDA Model、分页 MLA 块池和 Factory。同步请求与 SSE 请求都从文本 `prompt` 开始，经分词回调创建 Request，并在终态释放 Context、页表与服务条目。当前 Transport 用互斥锁串行调用 Adapter；因此该入口证明文本、设备执行和 HTTP/SSE 可以组合，不作为并发 HTTP batch 的性能证据。
