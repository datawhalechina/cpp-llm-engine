# 使用说明

这份文档帮助学习者选择路线、运行代码并判断学习结果。项目介绍见 [README.md](README.md)，参与修改见 [CONTRIBUTING.md](CONTRIBUTING.md)。

## 最短开始路径

默认 CPU 回归不下载模型，不要求网络、Python、CUDA 或外部 BLAS：

```bash
cmake --version
c++ --version
node --version
npm install
npm run test:regression
```

回归通过后，按以下顺序学习：

1. 运行 `bash code/chapter00_quickstart/run.sh`，确认 C++、CMake 和仓库入口可用。
2. 按 [Chapter00](source/chapter00_quickstart/intro.md) 准备 llama.cpp 与 TinyLlama，运行 `bash code/chapter00_quickstart/run_real_inference.sh` 建立真实推理基线。
3. C++、CMake 或测试基础不足时，按需学习 [Foundation](source/foundation/intro.md)。
4. 按 Chapter01–10 完成 CPU 主线；需要按阶段产物推进时，使用 [CPU 六任务学习计划](source/guides/cpu_learning_plan.md)。
5. CPU 基线稳定后，再选择 CUDA 或多模态扩展。

## 选择学习路线

CPU、GPU 和多模态共享同一套模型、执行和服务契约。章节导读中的 A/B 表示章内递进，G/M 表示在稳定契约上的横向扩展：

```text
A 教学基线 → B 真实主线
                  ├→ G GPU 执行扩展
                  └→ M 多模态输入扩展
```

| 路线 | 学习内容 | 建议起点 | 完成标志 |
| --- | --- | --- | --- |
| CPU 主线 | 从模型输入到单机推理服务 | [CPU 六任务学习计划](source/guides/cpu_learning_plan.md) | 默认回归通过，能解释权重到服务的完整链路 |
| GPU 扩展 | 将算子、工作区和 K/V 存储迁移到 CUDA | [GPU 六任务学习计划](source/guides/gpu_learning_plan.md) | CPU/CUDA 数值对照和真实设备测试通过 |
| 多模态扩展 | 增加图片预处理、Vision Encoder 和 Projector | [多模态六任务学习计划](source/guides/multimodal_learning_plan.md) | 图片形成 embedding 序列并进入现有 LLM Runtime |

A 建立可观察的小尺寸闭环，B 连接真实 CPU 结构；G 和 M 是选做扩展，不是排在 B 之后的必修章节。只学习执行核心时可先完成 Chapter00–06，继续学习缓存、调度和服务时再进入 Chapter07–10。

## 章节与运行入口

| 模块 | 学习产物 | 运行入口 |
| --- | --- | --- |
| Foundation | C++、构建、测试和 CUDA 前置能力 | `bash exercises/foundation/run_all.sh` |
| Chapter00 | 工具链检查与 TinyLlama 真实推理基线 | `bash code/chapter00_quickstart/run.sh` |
| Chapter01 | tokenizer、metadata、权重与 Model | `bash code/chapter01_tensor_loader/run.sh` |
| Chapter02 | 共享 Storage 的 Tensor view | `bash code/chapter02_tensor_ops/scripts/run.sh` |
| Chapter03 | 可验证的 CPU MatMul backend | `bash code/chapter03_matmul/scripts/run.sh` |
| Chapter04 | Attention、FFN 与单层 Decoder | `bash code/chapter04_attention/scripts/run.sh` |
| Chapter05 | Q4_K/Q6_K 量化 Linear | `bash code/chapter05_quantization/scripts/run.sh` |
| Chapter06 | Prefill、Decode、采样与流式输出 | `bash code/chapter06_generation/scripts/run.sh` |
| Chapter07 | BlockTable、分页 K/V 与缓存事务 | `bash code/chapter07_kv_cache/scripts/run.sh` |
| Chapter08 | Scheduler、Continuous Batching 与 Tensor batch | `bash code/chapter08_scheduling/scripts/run.sh` |
| Chapter09 | benchmark、profiling 与性能证据 | `bash code/chapter09_performance/scripts/run.sh` |
| Chapter10 | ServiceCore、HTTP/SSE 与 metrics | `bash code/chapter10_serving/scripts/run.sh` |

每章先读 `intro.md` 和正文，再从 `code/<chapter>/README.md` 进入实现、测试和运行脚本。

## 修改运行参数

示例程序和 benchmark 使用命令行参数改变请求、实验负载或已编译实现，不需要修改源码。先通过对应脚本完成构建，再在代码 README 中找到可执行程序和参数说明。

| 参数类型 | 示例 | 改变内容 |
| --- | --- | --- |
| 请求参数 | `--prompt`、`--max-new-tokens` | 单次生成输入与停止条件 |
| 负载参数 | `--batch`、`--iterations` | benchmark 的规模和采样次数 |
| 实现参数 | `--attention-path three-stage` | 从已编译且通过正确性验证的实现中选择一条 |

```bash
code/engine/build/cpp_llm_engine_demo \
  --prompt "hello" --max-new-tokens 4

/tmp/cpp-llm-chapter09-cuda-attention-build/real_tinyllama_cuda_attention_benchmark \
  llama.cpp/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf \
  --attention-path fused-online
```

CMake 选项决定编译哪些 backend，运行参数只选择已经编译的能力。实验编号和结果目录只用于保存证据，不会切换 kernel；未知参数会打印用法并退出。

## 可选扩展

| 扩展 | 入口 | 当前范围 |
| --- | --- | --- |
| CUDA backend | `CUDA_HOME=/usr/local/cuda-12.8 bash code/cuda/scripts/run.sh` | 设备资源、量化 Linear、K/V、GQA 和 Decoder |
| 真实模型集成 | [`code/engine/README.md`](code/engine/README.md) | TinyLlama CPU/CUDA Model、Runtime、Generation 和 Serving |
| 多模态 | `bash code/multimodal/scripts/run.sh` | SmolVLM CPU 输入、Vision、Projector 和 Decoder golden |

修改 CUDA backend、真实 CUDA Model/Context 或服务接入后，依次运行纵向验收：

```bash
CUDA_HOME=/usr/local/cuda-12.8 bash code/cuda/scripts/run.sh
bash code/engine/scripts/run_real_cuda_runtime.sh
bash code/engine/scripts/run_real_cuda_serving.sh
```

脚本默认读取 `llama.cpp/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf`，也可通过 `CPP_LLM_ENGINE_MODEL_PATH` 指定模型。各阶段的详细入口、成功条件和当前限制见 [`code/engine/README.md`](code/engine/README.md)；环境准备见 [CUDA 指南](source/guides/setup_cuda.md)，学习顺序见 [GPU 六任务学习计划](source/guides/gpu_learning_plan.md)。

## 判断验证结果

| 层级 | 入口 | 能够证明什么 |
| --- | --- | --- |
| 章节测试 | 对应章节 `run.sh` | 当前章节可以独立构建并通过测试 |
| 默认回归 | `npm run test:regression` | 无外部模型、网络和 GPU 时，主线资产与代码没有退化 |
| 发布契约 | `npm run test:release-contracts` | 全新 Release 构建中的测试登记、label 和跨章契约成立 |
| 文档检查 | `npm run docs:check-links` | 链接有效且 VitePress 站点可以完整构建 |
| 真实环境 | Engine、CUDA 或多模态对应脚本 | 指定模型、设备或输入路径真实运行通过 |

benchmark 只能比较固定环境和 workload 下的性能，不能替代正确性测试。实验条件与结果入口见[实验资产表](source/reference/experiment_assets.md)。

## 阅读代码的方法

1. 从章节 `intro.md` 确认上游输入、本章处理和下游输出。
2. 在正文中理解数据形状、公式、状态变化和设计理由。
3. 用 `code/<chapter>/README.md` 找到实现文件、测试和运行入口。
4. 先运行测试，再修改实现；优化时保留 CPU reference 或 golden。
5. 最后阅读框架对照，理解教学实现如何扩展到成熟系统。

阅读具体实现时，同时追踪三条线：

| 观察线 | 主要章节 | 需要回答的问题 |
| --- | --- | --- |
| 数据 | Chapter01–02、04、06–07 | 数据是什么形状、由谁持有、怎样随请求增长 |
| 算子 | Chapter03–05、CUDA | 一次计算做什么，由哪个 CPU/GPU backend 执行 |
| 系统 | Chapter00、06–10 | 请求怎样推进、测量、回收并通过服务输出 |

术语不明确时查[术语表](source/reference/glossary.md)，推导不明确时查[公式表](source/reference/formulas.md)，寻找跨章实现时查[教程资产索引](source/reference/tutorial_assets.md)。运行失败时先执行对应章节的 `run.sh`；Chapter00 环境和模型问题见[排错页](source/chapter00_quickstart/05_troubleshooting.md)。
