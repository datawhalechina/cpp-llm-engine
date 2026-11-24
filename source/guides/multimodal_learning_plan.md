# 多模态六任务学习计划

多模态路线在语言模型主线上增加图片输入。建议先完成 [CPU 六任务](./cpu_learning_plan.md)中的输入、算子和解码器任务，再按本计划把图片预处理、视觉编码器和投影层接入已有生成运行时。

学习计划负责学习顺序、运行入口和完成标准。SmolVLM、动态切图、SigLIP、Projector 与运行时交接的原理从[多模态扩展教程](../extensions/multimodal/intro.md)进入；能力状态和数值证据分别查[模型支持表](../reference/model_support.md)与[实验资产索引](../reference/experiment_assets.md#多模态cuda-与正确性资产)。

![多模态六任务学习路线](../assets/diagrams/multimodal-six-task-learning-route.svg)

## 总体顺序

| 顺序 | 任务 | 原理入口 | 完成后的可验证产物 |
| --- | --- | --- | --- |
| 1 | 固定模型与图片基线 | [扩展导读](../extensions/multimodal/intro.md) | 模型、mmproj、图片和外部输出身份可复现 |
| 2 | 完成图片预处理 | [图片预算与动态切图](../extensions/multimodal/01_image_preprocessing.md) | 图片变成形状和顺序确定的视觉输入 |
| 3 | 读取视觉模型 | [mmproj 与视觉编码器](../extensions/multimodal/02_vision_encoder.md) | 配置、张量布局和量化载荷通过检查 |
| 4 | 执行视觉编码器 | [mmproj 与视觉编码器](../extensions/multimodal/02_vision_encoder.md) | SigLIP Block 和 12 层编码器与独立结果一致 |
| 5 | 拼接统一嵌入 | [投影层与运行时交接](../extensions/multimodal/03_runtime_handoff.md) | 图片占位被视觉向量替换并进入语言模型 |
| 6 | 接入服务与性能证据 | [投影层与运行时交接](../extensions/multimodal/03_runtime_handoff.md) | 同步/流式输出、终态释放和重复测量通过 |

## 与主线章节的交接

多模态新增图片输入、视觉编码器、投影层和嵌入拼接，语言模型执行继续复用 Chapter01–10 的公共对象。

| 学习位置 | 多模态任务 |
| --- | --- |
| Chapter01–02 | 读取 mmproj，表达图片、Patch 和视觉嵌入 Tensor |
| Chapter03–05 | 执行视觉 Linear、Attention、Norm、FFN 和量化权重 |
| Chapter06–08 | 将统一嵌入交给生成运行时、K/V 和调度器 |
| Chapter09 | 固定视觉正确性、阶段耗时和性能工作负载 |
| Chapter10 | 接入同步/流式服务并检查终态释放 |

## CPU 首轮

首轮使用 CPU 逐步验证图片预处理、12 层 SigLIP、Projector、图文嵌入和语言解码器。默认组件测试不需要模型、网络或 GPU：

```bash
bash code/multimodal/scripts/run.sh
```

本地已有 SmolVLM、mmproj 和测试图片时，再运行真实 CPU 外部基线：

```bash
bash scripts/workflows/run_smolvlm_cpu_baseline.sh
```

模型路径、图片路径和逐项真实验证命令统一由 [`code/multimodal/README.md`](https://github.com/datawhalechina/cpp-llm-engine/blob/main/code/multimodal/README.md) 维护。

## GPU 进阶

GPU 路线复用语言模型的请求系统，只替换视觉数据面的执行位置。按下面顺序推进，每一步先与 CPU 结果比较：

| 顺序 | 学习对象 | 验证重点 |
| --- | --- | --- |
| 1 | Patch/Position Embedding 与 Projector | 权重布局、位置行收集和输出形状 |
| 2 | 单个 SigLIP 编码块 | 全注意力、LayerNorm、GELU 与双残差 |
| 3 | 12 层视觉编码器 | 跨层数值误差、显存峰值与持久工作区 |
| 4 | 完整 tile 到语言 Runtime | 1024 Patch、64 行视觉嵌入、24 个文本词元与四词元 CPU/CUDA 生成对照 |
| 5 | 服务生命周期与视觉批处理 | 先验证独立 CUDA Context、同步/SSE 和终态释放，再接图片传输与请求组批 |
| 6 | 分阶段性能证据 | 图片预处理、视觉编码、语言预填充和解码耗时 |

先运行 CUDA 组件测试：

```bash
CUDA_HOME=/usr/local/cuda-12.8 bash code/multimodal/scripts/run_cuda.sh
```

真实 mmproj、视觉流水线和语言 Context 的入口随代码能力更新，按代码 README 和[实验资产索引](../reference/experiment_assets.md#多模态cuda-与正确性资产)选择对应命令。

## 完成标准

| 检查项 | 完成条件 |
| --- | --- |
| 输入身份 | 保存模型、mmproj、图片 hash、Git commit 与运行环境 |
| 图片几何 | resize、padding、tile 顺序和 Patch shape 与固定输入一致 |
| 视觉执行 | CPU 与独立参考结果一致；CUDA 路径同时满足规定容差 |
| 嵌入交接 | 文本和视觉向量顺序正确，语言 Context 相互隔离 |
| 服务终态 | 同步、流式、失败和断连路径都释放请求资源 |
| 性能证据 | 固定工作负载、预热、原始样本和适用范围后再比较结果 |

完成首轮后，学习记录应能解释图片如何变成视觉词元、视觉词元如何进入语言模型，以及这条扩展复用了哪些 CPU/GPU 主线对象。
