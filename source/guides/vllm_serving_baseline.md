# vLLM Serving Baseline 部署服务框架基线

本指南用 `vLLM` 建立可选 serving baseline，验证服务启动、模型加载和 HTTP 文本返回。它不属于 Chapter00 的必读路径；先完成 `llama.cpp` 本地推理，再按需体验服务框架。

相关阅读：[`Chapter00 真实基线`](../chapter00_quickstart/intro.md)、[`vLLM 生态对照`](../reference/ecosystem/vllm.md)、[`10.2 准入与背压`](../chapter10_serving/next/02_admission_backpressure.md)、[`06.9 框架对照`](../chapter06_generation/next/09_framework_comparison.md)。
- 官方文档：[`Using Docker`](https://docs.vllm.ai/en/latest/deployment/docker/)、[`OpenAI-Compatible Server`](https://docs.vllm.ai/en/latest/serving/openai_compatible_server/)

## 运行前判断

运行本节需要：

- 你有可用 NVIDIA GPU 和 Docker 环境
- 已准备 Hugging Face 模型访问和缓存目录
- 希望体验 `request -> server -> model -> response` 的 OpenAI-compatible server

当前机器没有合适 GPU 或 Docker 时，跳过本指南，不影响 CPU 主线学习。

## 启动和请求

官方 `vllm/vllm-openai` 镜像可以直接启动 OpenAI-compatible server。下面使用一个较小的公开模型作为起步示例：

```bash
docker run --runtime nvidia --gpus all \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  --env "HF_TOKEN=$HF_TOKEN" \
  -p 8000:8000 \
  --ipc=host \
  vllm/vllm-openai:latest \
  --model Qwen/Qwen3-0.6B
```

几个参数先按字面理解：

- `--gpus all`：让容器能使用 NVIDIA GPU
- `-v ~/.cache/huggingface:/root/.cache/huggingface`：复用本机模型缓存
- `HF_TOKEN`：访问需要授权的 Hugging Face 模型时使用
- `-p 8000:8000`：把服务暴露到本机 `8000` 端口
- `--ipc=host`：给 PyTorch / vLLM 使用共享内存

如果你选择其他模型，只替换 `--model` 后面的模型名。模型越大，对显存要求越高。

### 请求验证

服务启动后，用另一个终端发起一次 Chat Completions 请求：

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-0.6B",
    "messages": [
      {"role": "user", "content": "Say hello in one sentence."}
    ],
    "max_tokens": 32
  }'
```

看到 JSON 返回，并且 `choices` 里有模型输出，就说明 serving baseline 已经跑通。

## 结果和后续

本次部署观察四件事：

- 服务以常驻进程运行
- 请求通过 HTTP 进入 runtime
- 模型加载后可以连续处理请求
- 输出以 OpenAI-compatible API 形式返回

这些现象会在后续章节被拆开：

| 你在 vLLM 里看到的现象 | 后续章节怎么拆 |
| --- | --- |
| 模型启动时加载权重 | `chapter01_tensor_loader` |
| 请求里的 token 进入模型 | `chapter06_generation` |
| 多请求需要排队和调度 | `chapter06_generation` |
| 长上下文需要复用 KV cache | `chapter06_generation` |
| 量化模型影响显存和速度 | `chapter05_quantization` |
| 服务性能需要 benchmark | `chapter09_performance` |

### 范围与失败处理

`PagedAttention`、continuous batching、block manager 和调度策略在 `chapter06_generation` 展开。本节启动失败时记录 GPU、镜像和模型加载错误；`llama.cpp` 首次推理通过后即可继续 Chapter01。
