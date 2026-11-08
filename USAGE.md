# 使用说明

这份文档帮助学习者选择路线、运行代码并判断验证结果。项目介绍见 [README.md](README.md)，参与修改见 [CONTRIBUTING.md](CONTRIBUTING.md)。

## 最短开始路径

默认 CPU 回归不下载模型、不要求网络，也不需要 Python、CUDA 或 BLAS：

```bash
cmake --version
c++ --version
node --version
npm install
npm run test:regression
```

回归通过后，按下面的顺序开始学习：

1. 运行 `bash code/chapter00_quickstart/run.sh`，确认 C++、CMake 和仓库入口可用。
2. 按 [Chapter00](source/chapter00_quickstart/intro.md) 准备 llama.cpp 与 TinyLlama，再运行 `bash code/chapter00_quickstart/run_real_inference.sh` 建立真实推理基线。
3. C++、CMake 或测试基础不足时，按需学习 [Foundation](source/foundation/intro.md)。
4. 按 Chapter 01 到 Chapter 10 完成 CPU 主线。
5. 需要按产物分解主线时，使用 [CPU 六任务学习计划](source/guides/phase1_cpu_learning_plan.md)。
6. CPU 基线稳定后，再选择 CUDA 或多模态扩展。

## 选择学习路线

三条完整路线共享同一套模型、执行和服务契约：CPU 建立完整基线，GPU 替换算子和设备内存实现，多模态增加图片到视觉向量的输入链路。章节导读中的 A/B 表示章内学习递进，G/M 表示在稳定契约上接入横向扩展。

### 章节中的路线标记

A 通常先建立可观察的教学基线，B 再进入真实结构或系统闭环；G 和 M 不排在 B 之后成为必修步骤，而是按学习目标选做：

```text
A 教学基线 → B 真实主线
                  ├→ G GPU 执行扩展
                  └→ M 多模态输入扩展
```

| 标记 | 所属维度 | 含义 |
| --- | --- | --- |
| A | 章内学习线 | 可观察的教学基线或第一阶段 |
| B | 章内学习线 | 真实结构、进阶实现或系统闭环 |
| G | 横向扩展线 | 在稳定契约上接入 GPU 执行 |
| M | 横向扩展线 | 在稳定主线上接入多模态输入 |

### 完整路线

只学习引擎执行核心时可以先完成 Chapter00–06；关注缓存、调度和服务时再继续 Chapter07–10。

| 路线 | 改变什么 | 复用什么 | 建议起点 | 完成标志 |
| --- | --- | --- | --- | --- |
| CPU 主线 | 从模型输入到推理服务建立完整实现 | 作为其他路线的数值与接口基线 | Foundation（按需）→ Chapter00 → Chapter01–10 | 默认回归通过，能解释从权重到服务的完整链路 |
| GPU 扩展 | 将算子、工作区和 K/V 存储迁移到 CUDA 设备 | Decoder、请求、调度和服务语义 | 完成 Chapter00–06 后进入 [`code/cuda`](code/cuda/README.md) | CPU/CUDA 数值对照和真实设备测试通过 |
| 多模态扩展 | 增加图片预处理、Vision Encoder 和 Projector | LLM Decoder、Runtime、K/V、调度和服务 | 完成 Chapter04–06 后进入 [`code/multimodal`](code/multimodal/README.md) | 图片形成统一 embedding 序列并进入现有 LLM Runtime |

## 章节与运行入口

| 模块 | 学习产物 | 运行入口 |
| --- | --- | --- |
| Foundation | C++、构建、测试和 CUDA 前置能力 | `bash exercises/foundation/run_all.sh` |
| Chapter00 | 工具链检查与 TinyLlama 真实推理基线 | `bash code/chapter00_quickstart/run.sh`<br>`bash code/chapter00_quickstart/run_real_inference.sh` |
| Chapter01 | tokenizer、metadata、权重与 Model | `bash code/chapter01_tensor_loader/run.sh` |
| Chapter02 | 共享 Storage 的 Tensor view | `bash code/chapter02_tensor_ops/scripts/run.sh` |
| Chapter03 | 可验证的 CPU MatMul backend | `bash code/chapter03_matmul/scripts/run.sh` |
| Chapter04 | Attention、FFN 与单层 Decoder | `bash code/chapter04_attention/scripts/run.sh` |
| Chapter05 | Q4_K/Q6_K 量化 Linear | `bash code/chapter05_quantization/scripts/run.sh` |
| Chapter06 | 单请求 Prefill、Decode、采样与流式输出 | `bash code/chapter06_generation/scripts/run.sh` |
| Chapter07 | BlockTable、分页 K/V 与缓存事务 | `bash code/chapter07_kv_cache/scripts/run.sh` |
| Chapter08 | Step Scheduler、Continuous Batching 与 Tensor batch | `bash code/chapter08_scheduling/scripts/run.sh` |
| Chapter09 | benchmark、profiling 与统一性能证据 | `bash code/chapter09_performance/scripts/run.sh` |
| Chapter10 | ServiceCore、HTTP/SSE 与 metrics | `bash code/chapter10_serving/scripts/run.sh` |

每章正文解释概念、数据流和设计取舍；`code/<chapter>/README.md` 列出该章代码资产；运行脚本负责构建并执行相应验证。

## 可选扩展

可选入口不属于默认 CPU 回归，应在满足环境条件后单独运行。

| 扩展 | 命令 | 当前范围 |
| --- | --- | --- |
| CUDA | `CUDA_HOME=/usr/local/cuda-12.8 bash code/cuda/scripts/run.sh` | [CPU/GPU 架构](source/reference/cpu_gpu_architecture.md)：设备资源、量化 Linear、连续/分页 K/V、GQA 和 Decoder；真实模型入口见 `code/engine/README.md` |
| 真实 CUDA 首层 | `bash code/engine/scripts/run_real_cuda_first_layer.sh` | TinyLlama layer-0 的 CPU/CUDA hidden 对照，以及 Model/Context 隔离 |
| 真实 CUDA Decoder | `bash code/engine/scripts/run_real_cuda_decoder.sh` | TinyLlama 2 层与 22 层的 CPU/CUDA hidden、分层 K/V 对照 |
| 真实 CUDA Runtime | `bash code/engine/scripts/run_real_cuda_runtime.sh` | 固定 token prompt 的 22 层 prefill、greedy decode 与请求隔离 |
| 真实 CUDA 文本生成 | `bash code/engine/scripts/run_real_cuda_generation.sh` | `Hello TinyLlama` 经 tokenizer、CUDA Runtime 和 detokenizer 得到 CPU 同款 golden |
| 真实 CUDA 服务 | `bash code/engine/scripts/run_real_cuda_serving.sh` | CUDA Executor 接入 Chapter10，同步 Completion、SSE 与四类终态释放通过 |
| Engine 集成 | `bash code/engine/scripts/run.sh` | 章节组件到统一 Engine 的集成与测试 |
| 真实 CPU 全链路验收 | `bash scripts/run_cpu_v1_acceptance.sh` | 兼容验收入口；要求本地 llama.cpp 和 TinyLlama GGUF |
| 多模态 | `bash code/multimodal/scripts/run.sh` | [多模态学习指南](source/guides/multimodal_inference.md)：SmolVLM CPU 输入、Vision、Projector 和 Decoder golden |

真实 TinyLlama CUDA 路径已连接 tokenizer、量化 embedding、22 层 Decoder、final norm、LM Head、Chapter06 Runtime、Chapter08 Decode batch、detokenizer 和 Chapter10 HTTP Adapter。Decode batch 合并 RMSNorm、Linear 与 FFN，并按精确 batch size 复用执行 workspace。CUDA 分页 payload、分页 GQA 和单请求真实 Context 已通过连续路径对照；分页 batch 事务与真实 socket server 仍在建设。

### GPU 纵向验收

修改 CUDA backend、真实 CUDA Model/Context 或服务接入后，依次运行下面三项。第一项验证独立 kernel、分页缓存和合成 Decoder；第二项验证 TinyLlama 真实 Runtime；第三项验证同一执行器进入同步 Completion、SSE 和终态释放。脚本默认读取 `llama.cpp/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf`，也可以通过 `CPP_LLM_ENGINE_MODEL_PATH` 指定模型。

```bash
CUDA_HOME=/usr/local/cuda-12.8 bash code/cuda/scripts/run.sh
bash code/engine/scripts/run_real_cuda_runtime.sh
bash code/engine/scripts/run_real_cuda_serving.sh
```

## 验证层级

不同命令提供不同强度的证据：

| 层级 | 命令或入口 | 证明范围 |
| --- | --- | --- |
| 章节测试 | 对应章节的 `run.sh` | 当前章节可以独立构建并通过测试 |
| 默认回归 | `npm run test:regression` | 无外部模型、网络和 GPU 时，检查静态资产并运行 Foundation、Chapter00–10、Engine 和多模态默认路径 |
| 发布契约 | `npm run test:release-contracts` | 用 Release 构建重新编译并运行已登记的关键契约，不下载模型、不运行 GPU 或 benchmark |
| 文档检查 | `npm run docs:check-structure`、`npm run docs:check-links`、`npm run docs:check-style`、`npm run docs:check-reference` | 文档结构、链接、风格和参考资产一致；链接检查会同时构建站点 |
| 代码头检查 | `npm run code:check-headers` | 教学代码文件具备用途、输入输出和验证说明 |
| 真实模型 | `CPP_LLM_ENGINE_REAL_MODEL=...` 对应的 Engine CTest | TinyLlama 权重、logits 或 token golden 成立 |
| GPU | `code/cuda/scripts/run.sh` | 当前 CUDA 资产在本机设备上构建和运行 |

benchmark 只能比较指定环境和 workload 下的性能，不能替代正确性测试。实验条件与结果登记规则见[实验资产表](source/reference/experiment_assets.md)。

`test:regression` 适合日常开发，它复用各章默认运行脚本。`test:release-contracts` 会在临时目录中使用 Release 配置重新构建关键 target，用于发布前确认测试登记、CTest label 和跨章契约仍然有效。

## 阅读代码的方法

1. 先读章节 `intro.md`，确认当前模块在完整引擎中的位置。
2. 再看章节正文中的数据形状、公式和最小伪代码。
3. 打开 `code/<chapter>/README.md`，找到实现文件、测试和运行入口。
4. 先运行测试，再修改实现；需要优化时保留 CPU reference 或 golden。
5. 最后阅读章节中的框架对照，理解教学实现如何扩展到成熟系统。

### 三条观察线

章节顺序用于连接从模型输入到推理服务的完整路径。阅读具体实现时，再分别追踪数据如何表示、计算如何执行，以及请求如何持续推进。

| 观察线 | 主要章节 | 重点对象 | 阅读时回答的问题 |
| --- | --- | --- | --- |
| 数据 | Chapter01–02、04、06–07 | GGUF、张量/视图、Q/K/V、上下文、块表 | 数据是什么形状、由谁持有、怎样随请求增长 |
| 算子 | Chapter03–05、CUDA | 矩阵乘、注意力、前馈网络、量化线性层、执行后端 | 一次计算做什么，由哪个 CPU/GPU 实现执行 |
| 系统 | Chapter00、06–10 | 运行时、调度器、基准测试、性能剖析、服务核心 | 请求怎样推进、测量、回收并通过 HTTP/SSE 输出 |

术语不明确时查[术语表](source/reference/glossary.md)，推导和符号不明确时查[公式表](source/reference/formulas.md)，查找跨章节实现时使用[章节资产表](source/reference/chapter_assets.md)。

## 常见问题

- `llama.cpp` 中的点是项目名称的一部分，不表示隐藏目录；仓库应位于项目根目录的 `llama.cpp/`。
- 模型文件、构建目录和下载缓存不要放入 Git 提交。
- 默认回归失败时，先运行失败章节的 `run.sh`，不要直接用真实模型或 GPU 路径排查。
- GPU 测试需要 `nvcc`、兼容驱动和可访问的 NVIDIA 设备；只通过编译不能证明设备执行正确。
- 修改文档后至少运行链接与结构检查；准备提交修改时继续阅读 [CONTRIBUTING.md](CONTRIBUTING.md)。
