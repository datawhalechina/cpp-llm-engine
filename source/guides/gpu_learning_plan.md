# GPU 六任务学习计划

GPU 路线在 CPU 已验证的算法和公共接口上替换设备内存、计算核与执行上下文。建议先完成 [CPU 六任务](./cpu_learning_plan.md)中的前四项；学习 GPU 调度和服务前，再完成 CPU 任务五、六。

TinyLlama 与 Qwen3 提供完整稠密 GPU 基线：真实模型加载、完整解码器、生成、分页 K/V、批量执行和服务均有验证入口。DeepSeek-V2-Lite 继续覆盖分页 MLA、权重吸收、MoE 专家执行、格式专用量化 GEMV 和设备端路由。三种模型复用同一生成运行时、调度和服务约定。

环境安装与故障排查见 [CUDA 安装与配置](./setup_cuda.md)，代码对象与测试清单见 [`code/cuda/README.md`](https://github.com/datawhalechina/cpp-llm-engine/blob/main/code/cuda/README.md)。

![GPU 六任务学习路线](../assets/diagrams/gpu-six-task-learning-route.svg)

## 总体顺序

| 顺序 | 任务 | 主要入口 | 完成后的可验证产物 |
| --- | --- | --- | --- |
| 1 | 固定 CUDA 环境 | Foundation CUDA、环境指南 | 驱动、工具包、编译器和 GPU 身份可记录 |
| 2 | 掌握设备资源与执行上下文 | `code/cuda` 资源层 | 缓冲区、CUDA 流、事件、cuBLAS 句柄和工作区生命周期正确 |
| 3 | 迁移核心算子 | Chapter03–05、`code/cuda` | 矩阵乘法、量化线性层、RoPE 和 GQA 与 CPU 结果一致 |
| 4 | 运行真实 GPU 解码器 | `code/model`、`code/engine` | TinyLlama 单层、2 层和 22 层结果通过 CPU/CUDA 对照 |
| 5 | 接入生成、批次与分页 K/V | Chapter06–08、`code/engine` | 真实生成、多请求批次和分页缓存正确推进 |
| 6 | 接入服务并建立性能证据 | Chapter09–10 | 同步/SSE 服务通过，GPU 优化有可重复的 A/B 证据 |

## 模型学习范围

首轮使用 TinyLlama 建立稠密模型的完整 GPU 路径，再按架构差异选择进阶模型。模型能力表负责记录可运行入口，实验资产表负责保存性能结论；学习计划只说明阅读和验证顺序。

| 模型 | 重点观察 | 建议任务 |
| --- | --- | --- |
| TinyLlama | 稠密解码器、GQA、分页 K/V 和完整服务链 | 任务 3–6 |
| Qwen3 | Q/K 归一化、NeoX RoPE 和第二种稠密配置 | 任务 4–6 |
| DeepSeek-V2-Lite | MLA、权重吸收、量化 MoE 和设备端路由 | 任务 3–6 的进阶实验 |
| SmolVLM | 视觉编码器、投影层与语言上下文交接 | 完成本计划后进入[多模态路线](./multimodal_learning_plan.md) |

多 GPU、分布式执行和多租户属于框架对照主题。模型支持范围和真实命令见[模型能力与扩展顺序](../reference/model_support.md)，性能实验编号和结论见[实验资产表](../reference/experiment_assets.md)。

## 共通验收记录

每项任务记录环境、输入、命令、正确性和资源变化，性能任务再保存原始样本。建议包含以下字段：

| 类别 | 需要记录的内容 |
| --- | --- |
| 环境 | 日期、Git commit、GPU、驱动、Toolkit、编译器和 CMake |
| 输入 | 模型、量化格式、提示词、批量和上下文长度 |
| 执行 | 命令、退出状态、CPU/CUDA 数值差异和关键输出 |
| 资源 | 峰值显存、工作区、CUDA 流、同步点和 K/V 布局 |
| 性能 | 预热次数、测量轮数、原始样本、统计量和工作负载范围 |
| 下一次学习 | 选择一个模型差异、一项计算核变化或一个待解释指标 |

## 任务 1：固定 CUDA 环境

先记录 GPU、驱动、CUDA 工具包（Toolkit）、编译器和 CMake，再运行最小设备访问检查。环境版本是后续数值和性能结果的一部分。

```bash
nvidia-smi
nvcc --version
cmake --version
c++ --version
```

完成标志是 `nvidia-smi` 与 `nvcc` 指向预期设备和工具包，`CUDA_HOME` 能被 CMake 识别。环境指南负责安装和路径排查，算子章节从可用工具链开始。

## 任务 2：掌握设备资源与执行上下文

阅读 `DeviceBuffer`、`PinnedHostBuffer`、`CudaStream`、`CudaEvent`、`CublasHandle`、`CudaWorkspace` 和 `CudaExecutionContext`。重点追踪谁拥有资源、异步操作在哪条 CUDA 流上发生、失败后由谁释放。

```bash
CUDA_HOME=/usr/local/cuda-12.8 bash code/cuda/scripts/run.sh
```

先关注资源、异步拷贝、事件计时和移动语义测试。编译成功只证明工具链可用；设备执行、错误返回和释放测试同时通过后，资源层才算完成。

## 任务 3：迁移核心算子

按“向量加法 → cuBLAS 矩阵乘法 → Q4_K/Q6_K 矩阵-向量乘法（GEMV）→ RoPE → GQA 注意力”阅读。每条 CUDA 路径都与 CPU 参考结果比较，再观察工作区、内存访问和融合策略。

| 对象 | 原理入口 | GPU 实现重点 |
| --- | --- | --- |
| 矩阵乘法 | Chapter03 | 行主序映射、cuBLAS 句柄与 CUDA 流 |
| 量化线性层 | Chapter05 | 分块解码、格式专用融合 GEMV 与批量权重复用 |
| RoPE、GQA、MLA | Chapter04 | 位置、注意力头映射、因果遮罩、低秩缓存与权重吸收 |
| MoE 路由器 | Chapter04 | 全专家 Softmax、稳定 Top-K、专家分发与设备路由 |

完成标志是 `code/cuda/scripts/run.sh` 中对应设备测试通过，并能解释 CPU 参考实现与 CUDA 计算核的输入输出为何相同。

## 任务 4：运行真实 GPU 解码器

核心算子通过后，把真实 TinyLlama 权重上传到 GPU，依次验证首层、短解码器和完整 22 层。模型保存共享权重，执行上下文保存请求位置与 K/V，两者的生命周期继续沿用 CPU 接口和状态规则。

```bash
CUDA_HOME=/usr/local/cuda-12.8 bash code/engine/scripts/run_real_cuda_first_layer.sh
CUDA_HOME=/usr/local/cuda-12.8 bash code/engine/scripts/run_real_cuda_decoder.sh
```

完成标志是 CPU/CUDA 隐藏状态、分层 K/V 和最终输出落在规定容差内，两个请求执行上下文相互隔离。

## 任务 5：接入生成、批次与分页 K/V

真实解码器进入 Chapter06 生成运行时后，继续验证预填充、逐词元解码、批量工作区和分页 K/V。调度器负责选择请求，GPU 执行器负责执行张量批次；请求终态与资源事务由 Chapter07–08 管理。

```bash
CUDA_HOME=/usr/local/cuda-12.8 bash code/engine/scripts/run_real_cuda_runtime.sh
CUDA_HOME=/usr/local/cuda-12.8 bash code/engine/scripts/run_real_cuda_generation.sh
CUDA_HOME=/usr/local/cuda-12.8 bash code/chapter09_performance/scripts/run_cuda_batch_ab.sh
```

完成标志是固定提示词产生 CPU 同款词元序列，连续两步能够复用工作区，分页 K/V 在成功时提交、失败时回滚。

## 任务 6：接入服务并建立性能证据

最后让同一个 GPU 执行器进入 Chapter10 的同步响应、SSE、取消和终态释放，再由 Chapter09 固定工作负载并记录性能结果。

```bash
CUDA_HOME=/usr/local/cuda-12.8 bash code/engine/scripts/run_real_cuda_serving.sh
CUDA_HOME=/usr/local/cuda-12.8 bash code/chapter09_performance/scripts/run_cuda_attention_ab.sh
```

性能实验同时保存 GPU、驱动、工具包、模型、提示词、输出长度、批量、原始样本和正确性结果。计算核结果解释局部变化，端到端 A/B 与热点归因共同决定优化是否适合作为默认路径。
