# 00.3 选择并准备真实模型

本教程使用 TinyLlama 1.1B Chat Q4_K_M 建立默认语言模型主线。它保留真实 Tokenizer、22 个 Decoder Block、RMSNorm、RoPE、GQA、SwiGLU、KV Cache 和 LM Head，同时把 GGUF 文件控制在约 638 MiB。第一次学习只下载 TinyLlama；Qwen3 和 Qwen1.5-MoE 分别在学习稠密架构适配与专家模型时按需准备。

## 选择理由

| 条件 | TinyLlama Q4_K_M 的作用 |
| --- | --- |
| CPU 可运行 | 没有 GPU 也能完成真实推理和后续回归 |
| 模型结构完整 | Chapter01–06 可以沿同一模型拆解加载、Decoder 和 Runtime |
| 量化格式真实 | Q4_K/Q6_K payload 可以进入 Chapter05 实验 |
| 外部结果可对照 | llama.cpp 和自研实现读取同一 GGUF |

TinyLlama 用于验证执行链，不用于代表更大模型的生成质量。

## 不同模型在教程中的角色

Chapter00 使用 llama.cpp 建立外部基线，后续章节再把同一模型接入教程自己的 C++ 执行链。当前自研 CPU 路径支持 TinyLlama、Qwen3-0.6B 和 Qwen1.5-MoE-A2.7B：TinyLlama 承担默认主线，Qwen3 验证第二种稠密架构，Qwen1.5-MoE 用于学习 Router、Top-K、共享专家和按专家分组执行。具体接入改动、资源限制和建设顺序见[模型基线与扩展顺序](../reference/model_support.md)。

| 模型 | 适合学习什么 | 教程当前状态 |
| --- | --- | --- |
| TinyLlama 1.1B Chat Q4_K_M | 从 GGUF、Decoder、生成到服务的完整主线 | 默认模型，CPU/CUDA 真实链路已接通 |
| Qwen3-0.6B | 权重命名、Tokenizer、聊天模板、Q/K Norm、NeoX RoPE 和 Decoder 配置适配 | CPU/CUDA 推理服务纵向闭环已接通 |
| DeepSeek-R1-Distill-Qwen-1.5B | 检查 Qwen 架构适配能否复用于另一套模型配置 | 排在 Qwen3-0.6B 之后；它不是 MoE |
| Qwen1.5-MoE-A2.7B-Chat | Router、Top-K、共享专家与专家执行 | CPU 24 层生成、批执行、调度与服务闭环已接通 |
| Qwen3-30B-A3B、DeepSeek-V2-Lite | 更大 MoE 与 MLA 工程差异 | 进阶对照，不属于第一阶段单卡自研运行承诺 |

## 下载与校验

从仓库根目录执行：

```bash
test -d llama.cpp/.git
mkdir -p llama.cpp/models
curl -L --fail --retry 3 -C - \
  -o llama.cpp/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf \
  https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
```

校验文件大小和 SHA256：

```bash
ls -lh llama.cpp/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
echo '9fecc3b3cd76bba89d504f29b616eedf7da85b96540e490ca5824d3f7d2776a0  llama.cpp/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf' \
  | sha256sum --check
```

校验通过后进入 [`第一次真实推理`](./04_first_inference.md)。下载中断时重复执行同一条 `curl -C -` 命令即可续传。

## 可选模型

Qwen3-0.6B 已接入教程 CPU/CUDA 主线，适合在完成 TinyLlama 后比较 Q/K Norm、NeoX RoPE 和非对称 Q/KV 投影。已有模型文件时通过环境变量切换：

```bash
CPP_LLM_ENGINE_MODEL_PATH="$PWD/llama.cpp/models/qwen3-0.6b/Qwen3-0.6B-Q4_K_M.gguf" \
bash code/chapter00_quickstart/run_real_inference.sh "Hello Qwen3" 4
```

Qwen1.5-MoE-A2.7B-Chat 用于后续 MoE 路线。下载脚本会先检查剩余磁盘空间，支持断点续传，并在完成后打印 SHA256：

```bash
bash code/chapter00_quickstart/download_qwen15_moe.sh
bash code/chapter00_quickstart/run_qwen15_moe_baseline.sh
```

这些 Chapter00 入口先建立外部结果。教程自己的实现与验证入口分布在 Chapter01、04～06、08～10，当前支持范围汇总在[模型基线与扩展顺序](../reference/model_support.md)。

## 验证与衔接

| 检查项 | 通过条件 | 失败时先看 |
| --- | --- | --- |
| llama.cpp 仓库 | `llama.cpp/.git` 存在 | 先完成 00.2 |
| GGUF 文件 | 大小约 638 MiB | 下载路径和剩余磁盘空间 |
| 文件内容 | `sha256sum --check` 输出 `OK` | 删除损坏文件后重新下载 |
