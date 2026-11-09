# GPU 六任务学习计划

GPU 路线在 CPU 已验证的算法和公共接口上替换设备内存、计算核与执行上下文。建议先完成 [CPU 六任务](./cpu_learning_plan.md)中的前四项；完整学习调度和服务前，再补齐 CPU 任务五、六。

当前真实 GPU 基线是 TinyLlama：单层、22 层 Decoder、Generation、分页 K/V、批量执行和服务均有验证入口。Qwen3 已完成 CPU 主线，GPU 路径仍待适配 Q/K Norm、NeoX RoPE、非对称 Q/KV 投影宽度和 28 层结构；因此任务 4–6 当前只承诺 TinyLlama。

环境安装与故障排查见 [CUDA 安装与配置](./setup_cuda.md)，代码对象与测试清单见 [`code/cuda/README.md`](/code/cuda)。

![GPU 六任务学习路线](../assets/diagrams/gpu-six-task-learning-route.svg)

## 总体顺序

| 顺序 | 任务 | 主要入口 | 完成后的可验证产物 |
| --- | --- | --- | --- |
| 1 | 固定 CUDA 环境 | Foundation CUDA、环境指南 | 驱动、Toolkit、编译器和 GPU 身份可记录 |
| 2 | 掌握设备资源与执行上下文 | `code/cuda` 资源层 | Buffer、Stream、Event、cuBLAS Handle 和 Workspace 生命周期正确 |
| 3 | 迁移核心算子 | Chapter03–05、`code/cuda` | MatMul、量化 Linear、RoPE 和 GQA 与 CPU 结果一致 |
| 4 | 运行真实 GPU Decoder | `code/model`、`code/engine` | TinyLlama 单层、2 层和 22 层结果通过 CPU/CUDA 对照 |
| 5 | 接入生成、批次与分页 K/V | Chapter06–08、`code/engine` | 真实生成、多请求批次和分页缓存正确推进 |
| 6 | 接入服务并建立性能证据 | Chapter09–10 | 同步/SSE 服务通过，GPU 优化有可重复的 A/B 证据 |

## 任务 1：固定 CUDA 环境

先记录 GPU、驱动、CUDA Toolkit、编译器和 CMake，再运行最小设备访问检查。环境版本是后续数值和性能结果的一部分。

```bash
nvidia-smi
nvcc --version
cmake --version
c++ --version
```

完成标志是 `nvidia-smi` 与 `nvcc` 指向预期设备和 Toolkit，`CUDA_HOME` 能被 CMake 识别。安装问题只在环境指南处理，不进入算子代码排查。

## 任务 2：掌握设备资源与执行上下文

阅读 `DeviceBuffer`、`PinnedHostBuffer`、`CudaStream`、`CudaEvent`、`CublasHandle`、`CudaWorkspace` 和 `CudaExecutionContext`。重点追踪谁拥有资源、异步操作在哪个 Stream 上发生、失败后由谁释放。

```bash
CUDA_HOME=/usr/local/cuda-12.8 bash code/cuda/scripts/run.sh
```

先关注资源、异步拷贝、事件计时和移动语义测试。编译成功只证明工具链可用；设备执行、错误返回和释放测试同时通过后，资源层才算完成。

## 任务 3：迁移核心算子

按 `Vector Add → cuBLAS MatMul → Q4_K/Q6_K GEMV → RoPE → GQA Attention` 阅读。每条 CUDA 路径都与 CPU 参考结果比较，再观察工作区、内存访问和融合策略。

| 对象 | 原理入口 | GPU 实现重点 |
| --- | --- | --- |
| MatMul | Chapter03 | 行主序映射、cuBLAS Handle 与 Stream |
| 量化 Linear | Chapter05 | 分块解码、融合 GEMV 与批量 SGEMM |
| RoPE、GQA | Chapter04 | 位置、Head 映射、因果遮罩与归约 |

完成标志是 `code/cuda/scripts/run.sh` 中对应设备测试通过，并能解释 CPU 参考实现与 CUDA 计算核的输入输出为何相同。

## 任务 4：运行真实 GPU Decoder

核心算子通过后，把真实 TinyLlama 权重上传到 GPU，依次验证首层、短 Decoder 和完整 22 层。模型保存共享权重，Context 保存请求位置与 K/V，两者的生命周期继续沿用 CPU 接口和状态规则。

```bash
bash code/engine/scripts/run_real_cuda_first_layer.sh
bash code/engine/scripts/run_real_cuda_decoder.sh
```

完成标志是 CPU/CUDA hidden、分层 K/V 和最终输出落在规定容差内，两个请求 Context 互不污染。

## 任务 5：接入生成、批次与分页 K/V

真实 Decoder 进入 Chapter06 Runtime 后，继续验证预填充、逐词元解码、批量工作区和分页 K/V。Scheduler 负责选择请求，GPU Executor 负责执行张量批次；请求终态与资源事务仍由 Chapter07–08 管理。

```bash
bash code/engine/scripts/run_real_cuda_runtime.sh
bash code/engine/scripts/run_real_cuda_generation.sh
```

完成标志是固定提示词产生 CPU 同款词元序列，连续两步能够复用工作区，分页 K/V 在成功时提交、失败时回滚。

## 任务 6：接入服务并建立性能证据

最后让同一个 GPU Executor 进入 Chapter10 的同步 Completion、SSE、取消和终态释放，再由 Chapter09 固定工作负载并记录性能结果。

```bash
bash code/engine/scripts/run_real_cuda_serving.sh
```

性能实验必须同时保存 GPU、驱动、Toolkit、模型、提示词、输出长度、批量、原始样本和正确性结果。单个计算核变快只能说明局部变化；是否保留优化由端到端 A/B 和热点归因共同决定。

## 学习验收记录

| 类别 | 需要记录的内容 |
| --- | --- |
| 环境 | 日期、Git commit、GPU、驱动、Toolkit、编译器和 CMake |
| 输入 | 模型、量化格式、提示词、批量和上下文长度 |
| 执行 | 命令、退出状态、CPU/CUDA 数值差异和关键输出 |
| 资源 | 峰值显存、工作区、Stream、同步点和 K/V 布局 |
| 性能 | warmup、iterations、原始样本、统计量和工作负载边界 |
| 结论 | 当前结果能证明什么、尚未运行什么、下一项单一变化 |
