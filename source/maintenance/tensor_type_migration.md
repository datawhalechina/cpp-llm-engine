# 跨章 Tensor 类型迁移路线

本规格用于维护 Chapter01–05、Core、CUDA 和 Engine 之间的数据表示。迁移目标是让跨章算子共享同一份布局契约，不要求模型文件、CPU Storage、CUDA Storage 和 token 序列使用同一个拥有型 C++ 类。

## 当前类型与处理决定

| 类型 | 当前职责 | 决定 | 理由 |
| --- | --- | --- | --- |
| Chapter01 `Tensor` | GGUF Tensor 名称、dtype、shape 和原始 payload | 保留 | 模型资产包含量化字节和文件语义，不等同于运行时激活 Tensor |
| Chapter02 `Tensor` | CPU Float32 Storage、View、shape、stride 和 offset | 作为 CPU 教学拥有型 Tensor | 已能导出公共 contract，适合承载 CPU 激活与跨章布局实验 |
| Chapter03 `Matrix` | 二维 Float32 MatMul 示例 | 退回教学外壳 | 二维字段便于讲 MatMul，但不应继续成为跨章接口 |
| Chapter04 `Tensor` | 二维 Decoder 教学计算 | 分阶段迁移 | 与 Chapter02 重复，且不能表达真实 Q/K/V 的 3D/4D 布局 |
| Chapter05 `Tensor` | INT8 教学量化与 scale | 保留为量化入门对象 | 真实 Q4_K/Q6_K 权重由专用 block 类型承载，不应伪装成普通激活 Tensor |
| Core `TensorContract` / `TensorViewRef` | 非拥有布局和只读数据引用 | 作为跨章接口 | 不绑定 Storage 类型、设备或章节实现 |
| CUDA `CudaTensor` | 连续 device Storage | 保留 | 设备内存所有权与 CPU `shared_ptr<vector<float>>` 不同，但可导出同一 contract |
| Engine `TokenTensor` | tokenizer 输出的 token ID 序列 | 保留并考虑改名 | 它是整数序列，不是数值算子 Tensor，不能参加激活 Tensor 合并 |

最终结构保持“多个拥有者，一个公共观察方式”：

```text
GGUF Weight / Chapter02 CPU Tensor / CudaTensor
                     ↓
          TensorContract + ViewRef
                     ↓
       MatMul / Attention / Linear Backend
```

## Adapter 的保留原则

Adapter 不是重复代码的同义词。负责以下转换时应保留：

- 文件 dtype、量化 block 到运行时权重视图；
- Host Storage 与 Device Storage 之间的显式复制；
- 模型布局到算子请求的语义映射，例如 GQA head mapping；
- 兼容旧教学 API 的短期外壳。

只做相同 CPU Float32 指针、shape 和 stride 搬运，并且上下游已经使用同一 contract 时，才可以删除 adapter。删除后，rank、dtype、device、连续性和越界检查必须仍有唯一归属，不能散落到调用方。

## 分阶段迁移

| 阶段 | 修改范围 | 本阶段不做 | 验证门槛 |
| --- | --- | --- | --- |
| M0 类型盘点 | 固定本规格与现有跨章测试 | 不改代码 | 文档结构、引用和风格检查通过 |
| M1 可写 View | Core 增加非拥有可写 Tensor View，并固定别名和范围检查 | 不增加拥有型 Core Tensor | Core CTest 覆盖空指针、越界、dtype 和原地输出 |
| M2 Chapter03 | MatMul 直接接收公共只读输入 View 和可写输出 View | 不删除 `Matrix` 示例 | Chapter02→03、Chapter03 和 Core 测试通过，数值保持 `58/64/139/154` |
| M3 Chapter04 基础算子 | 依次迁移 Linear、Norm、Residual、FFN | 不同时迁移 Attention 和完整 Decoder | 每迁移一个算子立即跑 Chapter04 固定输出和失败路径 |
| M4 Chapter04 Attention/Decoder | 用 Chapter02 3D/4D Tensor 或公共 View 承载 Q/K/V 和层输出 | 不引入 Tensor batch 并行 | causal、decode、batch 隔离、KV append 和真实 Decoder 验收通过 |
| M5 Chapter05 激活接口 | 量化 Linear 输入输出改用公共 View，量化权重保持专用类型 | 不统一 packed weight 与 Float32 Storage | reference、fused、AVX2、CUDA 与真实模型回归通过 |
| M6 Adapter 收口 | 删除无调用方且只转抄元数据的 adapter | 不删除设备、格式和模型语义转换 | 全局回归、真实 TinyLlama 和 CUDA 测试通过 |

M2–M5 每次只迁移一个可观察对象。接口、实现和测试不能在同一批中跨越两个阶段，否则失败时无法区分契约问题和数值问题。

当前 M3 已完成 Linear、LayerNorm、ReLU、Residual Add 和 FFN 的公共 View 入口。旧 Chapter04 Tensor API 继续作为教学兼容层，并反向复用新路径；每个算子均覆盖固定数值、错误 shape 不覆盖输出以及允许时的输入输出别名。M4 再迁移 Attention 和完整 Decoder，不在 M3 提前改变 Q/K/V 或 KV Cache 表示。

M4 已完成 3D/4D Attention、二维 MultiHeadAttention 和无缓存 DecoderLayer 的公共 View 入口。旧 Chapter02/04 Tensor 接口只在成功时提交结果。KV Cache 仍为独立拥有对象，尚未接入教学 DecoderLayer。

M5 已完成 Q4_K/Q6_K/Q8_0 packed 权重与 rank-1/2 Float32 激活 View 的适配。packed 权重继续使用 `QuantizedMatrixView`，不并入普通 Tensor；rank-2 当前逐 token 调用 backend，不表示并行 batch 执行。

## M6 Adapter 审计结果

本轮没有可安全删除的跨章 adapter：

| Adapter | 保留原因 | 可删除条件 |
| --- | --- | --- |
| Chapter03 MatMul View adapter | 校验二维 Float32 CPU 连续布局，并映射 `MatmulRequest` | backend 直接接收公共 View，且校验只有一个归属 |
| Chapter04 Attention View adapter | 映射 3D/4D、batch、head、decode offset 和逐 batch 请求 | Attention backend 原生接收相同 N 维契约 |
| Chapter05 Quantized Linear View adapter | 隔离 packed block 格式，映射 rank-1/2 激活和 policy | backend 原生接收 packed weight view 与激活 View |

返回 `Matrix` 或章节 Tensor 的兼容入口仍用于教学 API 和回归测试，并在内部复用公共 View 路径。后续只有在正文、示例和测试全部迁移到新入口后，才能单独删除兼容外壳；设备复制、量化格式和模型布局转换不属于删除对象。

## M1 可写 View 的最小契约

M1 已增加与只读 `TensorViewRef` 对称的 `MutableTensorViewRef`，使算子可以接收调用方提供的可写 Storage。两种 View 复用同一份布局校验：

| 字段或行为 | 要求 |
| --- | --- |
| Storage 指针 | 使用 `void*`，不拥有内存 |
| 布局 | 复用 `TensorContract` |
| 范围验证 | 与只读 View 使用同一 required-elements 计算 |
| 输入输出别名 | 每个算子显式声明是否允许 |
| 失败语义 | 校验失败时不得修改已有输出 |

Core 仍不负责分配内存。Chapter02 `Tensor` 已能导出只读或可写 CPU View；CUDA View 留在涉及设备接口的阶段实现。M2 的 Chapter03 MatMul 已接收公共只读输入和可写输出，`Matrix` 仅作为兼容教学外壳保留。

## 每步验证顺序

每一批代码修改按固定顺序执行：

1. 运行受影响模块的单元测试；
2. 运行相邻章节的跨章测试；
3. 运行 `bash scripts/run_regression_baseline.sh`；
4. 涉及真实 Decoder 时运行 `bash scripts/run_real_decoder_learning.sh`；
5. 涉及 CUDA View 或 backend 时运行 `CUDA_HOME=/usr/local/cuda-12.8 bash code/cuda/scripts/run.sh`。

只有新入口已被直接测试、旧入口仍通过且没有隐式 `contiguous()` 复制时，才能进入下一阶段。现有 [`Chapter02 N 维规格`](./chapter02_nd_tensor_design.md) 和 [`Chapter02→04 交接规格`](./chapter04_tensor_handoff_design.md) 继续作为 M2–M4 的局部设计输入。
