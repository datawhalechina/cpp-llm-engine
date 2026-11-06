# Chapter00–04 代码与文档证据矩阵

本表供维护者核对代码、正文和图片是否描述同一项能力。学习者正文只展示稳定的学习入口与通过条件，不展示建设状态和维护优先级。

## 使用规则

每项能力按以下顺序核对：

1. 代码对象已经存在，并且职责与正文描述一致。
2. 测试或真实模型脚本能够产生正文声明的结果。
3. 正文只写当前命令能够复现的能力。
4. 图片中的实线表示已有测试证据，虚线表示规划或外部参照。
5. 代码对象、章节依赖或验收命令变化时，同时复核本表、对应正文和 SVG。

静态文档检查只能证明文件和链接有效，不能证明图片语义与代码一致。本表中的“人工核对”不可由链接检查替代。

## 能力与证据

| 章节与能力 | 主要代码对象 | 默认或集成测试 | 真实模型 / CUDA 证据 | 学习者文档 | 图片 | 当前结论 |
| --- | --- | --- | --- | --- | --- | --- |
| Chapter00 工具链 smoke | `code/chapter00_quickstart/main_quick.cpp` | `bash code/chapter00_quickstart/run.sh` | 不适用 | `source/chapter00_quickstart/intro.md` | `chapter00-main.svg` | 默认 CPU 证据完整 |
| Chapter00 外部真实推理 | `run_real_inference.sh`、`llama-simple` | 不进入无模型回归 | `bash code/chapter00_quickstart/run_real_inference.sh` | `04_first_inference.md` | `chapter00-real-inference-chain.svg` | 真实连通性 smoke 完整；不声明固定 token golden |
| Chapter01 教学 Loader 与 Tokenizer | `Tensor`、`ModelLoader`、`Tokenizer` | `chapter01_loader_test`、`chapter01_schema_test`、`chapter01_tokenizer_payload_test` | 不适用 | `01_tensor_object.md` 至 `04_input_contract_and_tokenizer.md` | `chapter01-main.svg` | 默认 fixture 证据完整 |
| Chapter01 GGUF 配置与权重布局 | `GgufReader`、`ModelMetadataAdapter`、`GgufTensorLoader`、`ModelBuilder` | 默认 GGUF fixture 与 schema 错误测试 | `gguf_inspect <model> --validate-schema` | `05_model_metadata_and_gguf.md` | `chapter01-gguf-loading-chain.svg` | Reader 到 `LoadedModel` / `DecoderInput` 可核对 |
| Chapter01 真实 Tokenizer | `RealTokenizer`、`llama_tokenizer_compare` | 不进入无 llama.cpp 回归 | `bash code/chapter01_tensor_loader/run_real_tokenizer.sh` | `04_input_contract_and_tokenizer.md` | `chapter01-input-contract.svg` | 固定 TinyLlama token IDs 可复现；脚本与 Chapter00 共用 `build-cpu` 优先规则 |
| Chapter02 Tensor View | `Tensor`、`TensorContract`、`TensorViewRef` | Chapter02 三个 CTest | 不适用 | `source/chapter02_tensor_ops/` | `chapter02-main.svg`、`chapter02-view-materialization.svg` | 1–4 维、共享 Storage 和显式连续化证据完整 |
| Chapter03 CPU MatMul | `MatMulOp`、`CpuReferenceMatmulBackend`、`tensor_matmul_adapter` | Chapter03 三个 CTest | 不适用 | `source/chapter03_matmul/` | `chapter03-main.svg`、`chapter03-operator-backend-kernel.svg` | CPU 数值、失败保持和跨章 View 证据完整 |
| Chapter03 cuBLAS MatMul | `CublasMatmulBackend`、`cuda_cublas_matmul_test` | 不进入默认 CPU 回归 | `CUDA_HOME=... bash code/cuda/scripts/run.sh` | `04_cublas_implement.md` | `chapter03-operator-backend-kernel.svg` | 独立 CUDA target 已实现；不能画成 CPU Operator 的默认 dispatch |
| Chapter04 教学 Decoder | `MultiHeadAttention`、`LayerNorm`、`FeedForward`、`DecoderLayer` | Chapter04 三个 CTest | 可选 CUDA 算子对照 | `01_multihead_attention.md` 至 `04_decoder_layer.md` | `chapter04-main.svg`、`chapter04-attention-dataflow.svg`、`chapter04-decoder-attention-flow.svg` | 单层、3D/4D Attention 与公共 View 证据完整 |
| Chapter04/05 真实 TinyLlama Decoder | `RealLlamaCpuModel`、`RealLlamaCpuContext`、Chapter05 量化 Linear | 不进入默认无模型回归 | `bash scripts/run_real_decoder_learning.sh` | `05_load_first_layer.md`、`06_real_tinyllama_decoder.md` | `chapter01-tinyllama-decoder-hierarchy.svg` | 22 层 logits 指标和固定 greedy 序列可复现；Chapter04 L2 明确依赖 Chapter05 |

## 自动检查覆盖范围

| 检查 | 能发现 | 不能发现 |
| --- | --- | --- |
| `docs:check-structure` | 章节编号、侧边栏、维护文档泄漏和表格结构 | 能力声明是否真实 |
| `docs:check-links` | 图片、源码链接、生成页面和站点构建失败 | SVG 节点和箭头是否符合实现 |
| `docs:check-style` | 学习者口吻、禁用表达和基础风格问题 | 算法、shape 和执行顺序错误 |
| `test:regression` | 默认 CPU 代码和固定输出回归 | 外部模型、CUDA 和图片语义 |
| 真实模型脚本 | GGUF、Tokenizer、Decoder 和 token golden | 新环境的下载和构建目录是否与文档一致 |
| CUDA 脚本 | 设备执行与 CPU reference 数值对照 | 没有 GPU 的 CI 环境中的可用性 |

## 变更触发器

| 变更位置 | 必须复核 |
| --- | --- |
| `code/chapterXX` 的公开对象或脚本 | 同章 intro、相关正文、代码 README、本表 |
| `code/core` 的公共契约 | Chapter02–04 适配器、跨章测试、Tensor/Backend 图片 |
| `code/model` 或真实模型 golden | Chapter04 真实 Decoder 正文、层级图、实验资产 |
| `code/cuda` backend | Chapter03/04 CUDA 正文、Backend 图片、L3 验收描述 |
| 章节编号或依赖顺序 | 主图位置链、侧边栏、跨章链接、本表 |
| SVG 中的实现状态或固定数值 | 对应测试命令、结果资产和浏览器截图 |

提交前执行默认文档和 CPU 检查；真实模型或 CUDA 相关修改还要执行相应的 L2/L3 命令。图片最终审核继续遵循 `diagram_review_guidelines.md`。
