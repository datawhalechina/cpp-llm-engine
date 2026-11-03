# 章节文档与代码资产表

下表把每章的正文、代码、图、公式和验证入口放在同一行。开始学习前可用于定位材料，完成一章后可用于核对产物。

## Chapter 00-04：从真实基线到 Decoder

前五章建立真实输出基线，再依次得到可加载的模型、可共享 Storage 的 Tensor、可替换 Backend 的算子和可测试的 DecoderLayer。

| 章节 | 文档入口 | 代码入口 | 主图与公式 | 最小验证 |
| --- | --- | --- | --- | --- |
| Chapter00 真实推理起点 | [导读](../chapter00_quickstart/intro.md) | [代码资产](/code/chapter00_quickstart) | [主图](/diagrams/chapter00-main.svg)；本章以外部真实输出为准，不引入公式 | `bash code/chapter00_quickstart/run.sh` |
| Chapter01 输入、权重与模型配置 | [导读](../chapter01_tensor_loader/intro.md) | [代码资产](/code/chapter01_tensor_loader) | [主图](/diagrams/chapter01-main.svg)；[加载链](/diagrams/chapter01-gguf-loading-chain.svg)；[F-LOAD-01](./formulas.md#f-load-01) | `bash code/chapter01_tensor_loader/run.sh` |
| Chapter02 Tensor、Storage 与 View | [导读](../chapter02_tensor_ops/intro.md) | [代码资产](/code/chapter02_tensor_ops) | [主图](/diagrams/chapter02-main.svg)；[F-TENSOR-01/02](./formulas.md#f-tensor-01) | `bash code/chapter02_tensor_ops/scripts/run.sh` |
| Chapter03 Operator、MatMul 与 Backend | [导读](../chapter03_matmul/intro.md) | [代码资产](/code/chapter03_matmul) | [主图](/diagrams/chapter03-main.svg)；[执行分层图](/diagrams/chapter03-operator-backend-kernel.svg)；[F-LINEAR-01](./formulas.md#f-linear-01) | `bash code/chapter03_matmul/scripts/run.sh` |
| Chapter04 Attention、FFN 与 Decoder | [导读](../chapter04_attention/intro.md) | [代码资产](/code/chapter04_attention) | [主图](/diagrams/chapter04-main.svg)；[Decoder 数据流](/diagrams/chapter04-decoder-attention-flow.svg)；[F-ATTN-01](./formulas.md#f-attn-01) 至 [F-FFN-01](./formulas.md#f-ffn-01) | `bash code/chapter04_attention/scripts/run.sh` |

## Chapter 06-09：从生成 Runtime 到服务

后五章把 Decoder 放进请求生命周期，补上量化、可重复测量、性能归因和网络协议之外的 Service Core。

| 章节 | 文档入口 | 代码入口 | 主图与公式 | 最小验证 |
| --- | --- | --- | --- | --- |
| Chapter05 Quantization | [导读](../chapter05_quantization/intro.md) | [代码资产](/code/chapter05_quantization) | [主图](/diagrams/chapter05-main.svg)；[05→06 接口交接](/diagrams/chapter05-06-execution-handoff.svg)；[Q4_K Backend](/diagrams/chapter05-q4k-backend.svg)；[F-QUANT-01/02](./formulas.md#f-quant-01) | `bash code/chapter05_quantization/scripts/run.sh` |
| Chapter06 Generation Runtime | [导读](../chapter06_generation/intro.md) | [代码资产](/code/chapter06_generation) | [主图](/diagrams/chapter06-main.svg)；[Runtime 状态图](/diagrams/chapter06-runtime-state.svg)；[F-KV-01](./formulas.md#f-kv-01)、[F-SCHED-01](./formulas.md#f-sched-01)、[F-SAMPLE-01](./formulas.md#f-sample-01) | `bash code/chapter06_generation/scripts/run.sh` |
| Chapter07 Benchmark 与 Correctness | [导读](../chapter07_benchmark/intro.md) | [代码资产](/code/chapter07_benchmark) | [主图](/diagrams/chapter07-main.svg)；[F-PERF-01/02](./formulas.md#f-perf-01) | `bash code/chapter07_benchmark/scripts/run.sh` |
| Chapter08 Profiling 与 Optimization | [导读](../chapter08_profiling/intro.md) | [代码资产](/code/chapter08_profiling) | [主图](/diagrams/chapter08-main.svg)；[真实服务分解图](/diagrams/chapter08-real-service-profile.svg)；[F-PERF-01/02](./formulas.md#f-perf-01) | `bash code/chapter08_profiling/scripts/run.sh` |
| Chapter09 Serving Extensions | [导读](../chapter09_serving/intro.md) | [代码资产](/code/chapter09_serving) | [主图](/diagrams/chapter09-main.svg)；[延迟时间线](/diagrams/chapter09-latency-timeline.svg)；[F-SERVE-01](./formulas.md#f-serve-01) | `bash code/chapter09_serving/scripts/run.sh` |

## 跨章节扩展资产

GPU、Engine 和多模态复用 Chapter01-09 的公共契约，不另建一套主线章节。它们的资产入口和默认依赖条件如下。

| 扩展 | 文档入口 | 代码入口 | 图与公式 | 默认验证 |
| --- | --- | --- | --- | --- |
| 真实模型执行 | [Chapter04 真实 Decoder](../chapter04_attention/06_real_tinyllama_decoder.md) | [Model 资产](/code/model) | [TinyLlama 层级图](/diagrams/chapter01-tinyllama-decoder-hierarchy.svg) | `bash scripts/run_real_decoder_learning.sh` |
| 公共契约与 Engine 组装 | [执行分层](./execution_layers.md) | [Engine 与 core](/code/engine) | [代码架构图](/diagrams/code-architecture.svg) | `npm run test:regression` |
| CUDA 执行扩展 | [CPU/GPU 架构](./cpu_gpu_architecture.md) | [CUDA 资产](/code/cuda) | [CPU/GPU 路线图](/diagrams/cpu-gpu-learning-route.svg)；[F-CUDA-01](./formulas.md#f-cuda-01) | CUDA 可用时按 `code/cuda/README.md` 运行 |
| 多模态输入扩展 | [多模态输入链路](./multimodal_inference.md) | [多模态资产](/code/multimodal) | [Runtime 阶段图](/diagrams/multimodal-runtime-stages.svg)；[F-MM-01](./formulas.md#f-mm-01) 至 [F-MM-03](./formulas.md#f-mm-03) | `cmake`、`ctest` 默认 fixture 闭环 |
