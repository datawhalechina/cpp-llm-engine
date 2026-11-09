# vLLM 服务快速体验

本指南通过独立 Docker 容器运行一次 vLLM 服务，让学习者先观察“常驻进程接收 HTTP 请求并调用模型”的完整形态。容器与本教程 C++ 代码彼此独立；完成体验后，再从 Chapter07–10 学习缓存、调度、性能证据和服务实现。

vLLM 的内部工程对象与源码入口见[生产级推理框架对照](../reference/ecosystem/production_frameworks.md#vllm)。以下命令来自 vLLM 官方的 [Docker 部署](https://docs.vllm.ai/en/stable/deployment/docker/)和 [OpenAI 兼容服务](https://docs.vllm.ai/en/stable/serving/openai_compatible_server/)说明。

## 运行条件

准备以下环境：

- 可由 `nvidia-smi` 访问的 NVIDIA GPU
- Docker 与 NVIDIA Container Toolkit
- 足够的模型下载空间和可选的 Hugging Face 访问令牌

```bash
nvidia-smi
docker --version
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu24.04 nvidia-smi
```

CPU 主线不依赖这些条件。当前机器无法运行 GPU 容器时，可以直接继续 Chapter01–10。

## 启动服务

下面沿用 vLLM 官方示例中的 Qwen3-0.6B。它只用于体验外部服务，不改变本教程以 TinyLlama 为默认模型的实现主线。

```bash
docker run --rm --name cpp-llm-vllm \
  --runtime nvidia --gpus all \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  --env "HF_TOKEN=$HF_TOKEN" \
  -p 8000:8000 \
  --ipc=host \
  vllm/vllm-openai:latest \
  --model Qwen/Qwen3-0.6B
```

第一次运行会下载镜像和模型。日志显示模型加载完成、服务监听 `8000` 端口后，再打开另一个终端发送请求。

`latest` 适合首次体验。需要保存可重复实验时，应固定镜像标签、模型 revision、GPU、驱动和请求参数，并按 Chapter09 的实验资产规则登记结果。

## 发送同步请求

先查看服务公开的模型：

```bash
curl http://localhost:8000/v1/models
```

再发送一次 Chat Completions 请求：

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-0.6B",
    "messages": [
      {"role": "user", "content": "用一句中文介绍推理服务。"}
    ],
    "max_tokens": 32
  }'
```

响应中的 `choices[0].message.content` 应包含非空文本。这个结果验证模型加载、HTTP 请求解析和同步文本返回已经连通，不用于比较回答质量。

## 观察流式输出

将 `stream` 设为 `true`，观察服务持续返回事件：

```bash
curl -N http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-0.6B",
    "messages": [
      {"role": "user", "content": "依次输出三个数字。"}
    ],
    "max_tokens": 16,
    "stream": true
  }'
```

终端会收到多条 `data:` 事件，最后出现结束事件。Chapter10 将用本教程自己的 `ResponseWriter` 和 HTTP 适配器解释相同的流式交付问题。

体验完成后停止容器：

```bash
docker stop cpp-llm-vllm
```

## 与教程章节的关系

| 在 vLLM 中观察到的现象 | 本教程的学习位置 |
| --- | --- |
| 服务启动时加载模型和权重 | Chapter01 输入、配置与权重 |
| 请求经过预填充和逐词元解码 | Chapter06 生成运行时 |
| 长上下文占用并复用 K/V 块 | Chapter07 K/V Cache |
| 多请求组成连续批次 | Chapter08 调度与批处理 |
| 延迟、吞吐和热点需要可重复证据 | Chapter09 性能证据 |
| HTTP、流式输出、取消和资源释放 | Chapter10 推理服务 |

这项体验证明外部 vLLM 容器可以提供模型服务。本教程的 C++ Loader、Decoder、Scheduler 和 ServiceCore 仍由仓库自身实现；需要比较两套服务的性能时，应先在 Chapter09 固定相同模型、输入、并发、输出长度和设备条件。
