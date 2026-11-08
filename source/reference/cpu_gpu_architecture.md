# CPU 主线与 GPU 扩展架构

教程先用 CPU 固定正确性和接口，再在同一套公共契约上接入 GPU。GPU 不是另一条 Runtime，而是替换 Backend、Kernel 和内存执行路径。

多模态不是第三种计算后端：它在 Decoder 前增加图片、Vision Encoder 和 Projector，再把统一 embedding 序列交给现有 CPU 或 GPU 执行路径。图片怎样进入 Runtime 见[多模态推理输入链路](./multimodal_inference.md)。

![推理引擎 CPU 主线与 GPU 扩展架构](../assets/diagrams/inference-engine-cpu-gpu-architecture.svg)

## 读图顺序

| 顺序 | 观察对象 | 学习问题 |
| --- | --- | --- |
| 1 | 输入与权重 | tokenizer、GGUF 和 Tensor 怎样进入引擎 |
| 2 | Core Contract | shape、dtype、device 和 ownership 怎样跨后端保持一致 |
| 3 | Operator / Decoder / Runtime | 算子、模型结构和请求推进怎样分工 |
| 4 | CPU / GPU Backend | 同一算子怎样选择不同计算实现 |
| 5 | Benchmark / Profiling | 怎样证明行为正确、性能变化真实且原因可解释 |

## 两条后端路径

当前 GPU 公共层已经包含资源 RAII、<a href="../_code/cuda/include/cpp_llm_engine/cuda/cuda_tensor.h"><code>CudaTensor</code></a>、可复用 Workspace、ExecutionContext、Vector Add、cuBLAS MatMul、单 token Q4_K/Q6_K 融合 GEMV、批量反量化加 SGEMM、RoPE、连续/分页 device K/V 和 causal GQA Attention。ExecutionContext 接管 stream、handle 和临时显存，上层算子不直接调用 `cudaMalloc/cudaFree`。

| 层 | CPU 主线 | GPU 扩展 |
| --- | --- | --- |
| Backend | reference、fused、SIMD、线程并行 | CUDA、cuBLAS、量化 Linear、RoPE、三阶段 GQA；fused quant/Attention 待优化 |
| Memory / Execution | host Tensor、连续/分页 K/V、workspace | `CudaTensor`、`CudaQuantizedMatrix`、`CudaWorkspace`、`CudaExecutionContext`、连续与分页 K/V payload |
| 主要目标 | 正确性、调试、回归和 CPU First Success | 带宽、吞吐、并行度和服务规模 |
| 不改变的部分 | `Operator`、`Decoder`、`Request`、`Scheduler` 的上层语义 | 同左，除非设备异步语义要求显式状态 |

## 分层职责

CPU 与 GPU 可以共享上层语义，是因为每一层只承担一种主要职责。GPU 接入从 Backend、Kernel 和设备内存开始，不重新实现模型结构或请求生命周期。

| 层 | 共同职责 | CPU 实现 | GPU 实现 | 不负责 |
| --- | --- | --- | --- | --- |
| Core Contract | Shape、Status、ownership 与 Decoder 执行语义 | 固定跨章接口 | 保持同一语义 | 不实现 kernel |
| Operator | MatMul、Linear、Attention 的输入输出契约 | 调用 CPU backend | 调用 CUDA backend | 不管理请求 |
| Backend | 选择具体执行路径 | reference、fused、SIMD、线程 | cuBLAS、自定义 CUDA kernel | 不定义模型结构 |
| Kernel | 完成局部数值计算 | scalar、AVX2、线程并行 | GEMV、GQA、RoPE、归约与融合 | 不处理 tokenizer |
| Decoder | 组合 Embedding、Attention、FFN 和 K/V | CPU 可运行实现 | 复用结构并替换计算路径 | 不决定请求顺序 |
| Runtime | Prefill、Decode、Sampling 和 Request | 完整主线 | 复用 Request/Runtime | 不直接解析 GGUF |

## 策略放在 CPU 还是 GPU

请求选择、资源所有权、状态提交和失败回滚先形成平台无关或 CPU 可运行语义；大规模张量计算、设备内存访问和 kernel 融合再由 CUDA backend 实现。CPU reference 不模拟 warp、shared memory 或 Tensor Core，只提供可读算法和数值基线。

| 策略 | CPU 或平台无关部分 | GPU 部分 | 当前状态 |
| --- | --- | --- | --- |
| Continuous Batching | Step 计划、token/block budget、终态和回滚 | 执行组好的 Tensor batch | CPU 与真实 CPU/CUDA batch 已接通 |
| Paged K/V | BlockTable、预留、提交、裁剪和回收 | 分页 payload 与按页表读取的 GQA | CPU/CUDA 单请求和双请求 batch 已验证 |
| FlashAttention | 三段式 golden 与分块在线 Softmax 参考 | tile、片上存储、Warp 归约与融合 kernel | CUDA v2、22 层 token Tensor Prefill 和 Runtime golden 通过；端到端 A/B 收益不稳定，默认仍为三阶段；分页 Prefill 未完成 |
| Radix Cache | 前缀树、最长匹配、共享块引用和释放 | 复用命中页表读取 K/V | CPU 最小闭环与 Chapter08 缓存感知排序已实现；GPU 复用待接入 |
| MTP / 推测解码 | 候选、接受规则和状态回滚 | 并行验证多个候选词元 | 尚未实现 |
| MLA | 压缩 K/V 和位置编码分支的数值参考 | 矩阵计算与专用 Attention kernel | 随 DeepSeek 模型路线规划 |
| Prefill/Decode 分离 | 实例角色、路由和 K/V 传输协议 | 各实例的 Prefill/Decode 执行 | 属于分布式扩展，当前不实现 |

## 实现顺序

1. 先用 CPU reference 固定输入输出和错误行为。
2. 用 CPU fused / SIMD 优化，但保留 reference 作为数值基线。
3. 用 benchmark 固定 workload、正确性和阶段指标。
4. 再接 CUDA / cuBLAS，验证 GPU 结果与 CPU reference 一致。
5. 最后用 profiling 解释 kernel、带宽、同步和调度成本。

这条顺序避免同时改变算子契约、数据布局和设备执行模型。

## 章节实施顺序

| 阶段 | 对应章节 | 可验证产物 |
| --- | --- | --- |
| 输入和对象 | Chapter01–02 | GGUF schema、Tensor view、共享 storage |
| 计算契约 | Chapter03 | Operator、CPU Backend、MatMul 与 Linear |
| 模型执行 | Chapter04 | DecoderLayer、Attention、RoPE 与连续 K/V |
| 量化路径 | Chapter05 | Q4_K/Q6_K/Q8_0 reference、fused Linear 与 CUDA GEMV |
| 生成闭环 | Chapter06 | Prefill、Decode、Sampling 与 Request |
| 缓存与调度 | Chapter07–08 | 分页 K/V、Radix Cache、Step Scheduler 与 Tensor batch |
| 证据闭环 | Chapter09 | benchmark、profiling、A/B 和回归基线 |
| 服务闭环 | Chapter10 | ServiceCore、HTTP/SSE、并发和终态释放 |

## 公共契约怎样接入

`code/core` 提供跨章契约，不要求各章一次性改写自己的教学对象。迁移按以下顺序进行：

```text
先写 core contract test
  -> 章节实现适配 contract
  -> 保留章节 correctness test
  -> 增加跨章 integration test
  -> GPU backend 复用同一 contract
```

| 对象 | 公共契约与当前接入 |
| --- | --- |
| Tensor view | Chapter02 提供 shape、stride、storage offset、dtype 与 device contract |
| MatMul | Chapter03 通过 `MatmulRequest`、`MatmulBackend` 和 `Status` 接入 CPU reference |
| Quantized Linear | Chapter05 通过 `LinearRequest`、backend policy 接入 scalar、AVX2 与 CUDA 路径 |
| Decoder 结构 | Chapter04 教学层和 `code/model` 真实模型共同适配 `DecoderLayerContract` |
| Decoder 执行 | CPU/CUDA Context 适配 `DecoderExecutionContext`，保持请求级位置和 K/V 独立 |
| Runtime 资源 | Chapter07–10 通过事务、BatchRunner 和 ServiceCore 交接，不让 kernel 拥有请求状态 |

## 验证入口

只核对公共契约时运行 Core 测试；核对所有 CPU 章节、CUDA 可选资产声明和文档结构时运行全局回归。设备数值和性能实验使用各章 CUDA/Chapter09 入口，不混入无 GPU 的默认回归。

```bash
cmake -S code/core -B /tmp/cpp-llm-engine-core-build
cmake --build /tmp/cpp-llm-engine-core-build
ctest --test-dir /tmp/cpp-llm-engine-core-build --output-on-failure

bash scripts/run_regression_baseline.sh
```
