# 06.0 构建单请求生成核心

## 本章导读

Chapter04 已经固定 Decoder 的计算顺序，Chapter05 让真实量化权重能够进入 Linear。本章把一次模型前向扩展为单请求生成循环：先用 Prefill 处理 prompt，再逐词元 Decode，经过采样、停止判断和回调得到可观察输出。Chapter07 在这组语义下管理模型 K/V，Chapter08 再选择多个请求怎样共享执行机会。

| 学习线 | 本章怎样学习 | 可验证产物 |
| --- | --- | --- |
| A 教学闭环 | 用小词表、确定性 logits 和 Greedy 跟踪请求状态 | 固定 token、阶段、停止条件和回调顺序 |
| B 真实执行闭环 | 接入 TinyLlama Tokenizer、22 层 Decoder 和请求专属模型执行上下文 | 固定生成 token、文本结果和请求状态隔离 |
| G GPU 扩展坐标 | 保持 Request 和停止语义不变，将模型执行器替换为设备后端 | 真实文本已完成 CUDA prefill、greedy decode、detokenize 和 Context 隔离 |
| M 多模态扩展坐标 | 视觉 embedding 进入模型后复用 Decode、采样和终态语义 | 当前接口坐标，不作为本章可运行输入路径 |

| 本章新术语 | 先这样理解 | 首次解释 |
| --- | --- | --- |
| 预填充（`prefill`） | 一次处理 prompt，并建立后续 Decode 所需的模型状态 | [`06.2 Prefill 与 Decode`](./02_prefill_decode.md) |
| 逐词元解码（`decode`） | 每次提交一个新词元并产生下一组 logits | [`06.2 Prefill 与 Decode`](./02_prefill_decode.md) |
| 模型执行上下文（Context） | 保存或绑定一个请求的位置、分层 K/V 和后端状态 | [`06.1 请求与执行契约`](./01_request_contract.md) |
| 模型执行器（`RequestModelExecutor`） | 使用一个请求的 Context，回答 logits 是什么 | [`06.1 请求与执行契约`](./01_request_contract.md) |
| 执行器工厂（Factory） | 复用共享模型，并为请求创建执行器和 Context | [`06.1 请求与执行契约`](./01_request_contract.md) |
| 采样器（`Sampler`） | 从 logits 选择下一个词元 | [`06.5 采样与停止`](./05_sampling_stop.md) |

![Chapter 06 主图：单请求怎样完成 Prefill、Decode 和输出](../../assets/diagrams/chapter06-generation-core.svg)

## 开始学习

先沿单请求的状态推进理解 Generation Core：请求保存可观察状态，模型执行器持有模型执行上下文，Runtime 组织 Prefill、Decode、采样、停止和回调。随后再用真实 TinyLlama 替换教学执行器，观察控制流保持不变。

| 顺序 | 路线 | 页面 | 先看源码 | 读完后应看到什么 |
| --- | --- | --- | --- | --- |
| 1 | A/B 公共 | [`06.1 请求与执行契约`](./01_request_contract.md)–[`06.3 单请求循环`](./03_single_request_loop.md) | <a href="../../_code/chapter06_generation/src/runtime.cpp"><code>runtime.cpp</code></a> | prompt 怎样经过 Prefill 和 Decode 产生输出 |
| 2 | A/B 公共 | [`06.4 请求状态`](./04_request_lifecycle.md) | <a href="../../_code/chapter06_generation/src/request.cpp"><code>request.cpp</code></a> | 五种终态为什么不能再次推进 |
| 3 | A | [`06.5 采样与停止`](./05_sampling_stop.md) | <a href="../../_code/chapter06_generation/src/sampler.cpp"><code>sampler.cpp</code></a> | 固定 logits 怎样得到固定词元并结束请求 |
| 4 | A/B 公共 | [`06.6 单请求 K/V 语义`](./06_single_request_kv_semantics.md) | <a href="../../_code/chapter06_generation/include/chapter06_generation/model_executor.h"><code>model_executor.h</code></a> | 词元历史与模型分层 K/V 为什么不是同一对象 |
| 5 | A/B 公共 | [`06.7 流式与错误`](./07_streaming_errors.md) | <a href="../../_code/chapter06_generation/src/runtime.cpp"><code>runtime.cpp</code></a> | 词元提交、回调和失败状态的先后顺序 |
| 6 | B | [`06.8 真实 TinyLlama`](./08_real_tinyllama_generation.md)、[`06.9 框架对照`](./09_framework_comparison.md) | `real_tinyllama_generation_smoke.cpp` | 真实 token golden 与成熟 Runtime 的对应关系 |

## 验证与下一步

完成本章后，一个请求可以独立完成 Prefill、Decode、采样、停止和逐词元通知；后续章节可以改变 K/V 存储或请求选择，而不改变这组结果语义。

| 验收层级 | 执行入口 | 通过条件 | 本章要求 |
| --- | --- | --- | --- |
| L1 默认 CPU | `bash code/chapter06_generation/scripts/run_generation_core.sh` | 单请求生成、Greedy、终态和回调测试通过，示例输出 `request phase: finished` | 必做 |
| L2 真实模型 | `bash code/chapter06_generation/scripts/run_real_generation.sh` | token golden 和连续请求 Context 隔离通过 | 有模型时必做 |
| L3 可选 CUDA | `bash code/engine/scripts/run_real_cuda_generation.sh` | `Hello TinyLlama` 生成 CPU/CUDA 相同的四个 golden token，重复请求结果一致 | 有 NVIDIA GPU、模型和 llama.cpp 时选做 |
