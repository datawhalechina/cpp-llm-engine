# Chapter 00：Quickstart 真实推理起点

本目录验证外部真实推理基线，不实现教程自己的 Tensor、Decoder 或 Runtime。完整学习说明见 [`source/chapter00_quickstart`](../../source/chapter00_quickstart/intro.md)。

## 代码架构

Chapter00 有两条互相独立的验证路径：仓库内最小程序检查本地工具链，外部 `llama.cpp` 加载 TinyLlama 建立真实推理基线。前者失败时排查 C++ 和 CMake；前者通过而后者失败时，排查外部构建、模型或运行参数。

```text
run.sh -> CMake -> chapter00_quickstart --check -> 连通性检查通过

run_real_inference.sh -> llama-simple + TinyLlama GGUF -> generated tokens
```

本目录不实现 Tensor、Decoder 或 Runtime，也不会联网下载源码和模型。

## 阅读顺序与资产

先读 `run.sh` 和 `examples/main_quick.cpp`，理解无模型检查；再读 `run_real_inference.sh`，观察真实模型路径、推理参数和结果检查。

| 资产 | 作用 |
| --- | --- |
| `CMakeLists.txt` | 构建本章连通性检查程序 |
| `examples/main_quick.cpp` | 检查模型路径和 quickstart 入口 |
| `run.sh` | 从临时目录配置、构建并运行连通性检查 |
| `run_real_inference.sh` | 检查本地 llama.cpp 与 GGUF，并执行一次真实推理 |
| `tests/run_real_inference_contract.sh` | 用伪执行程序验证脚本成功与失败分支 |

## 运行与验证

从仓库根目录执行：

```bash
bash code/chapter00_quickstart/run.sh
```

该命令验证连通性检查程序和真实推理脚本契约，不读取真实模型。准备好 llama.cpp 和 TinyLlama 后，一条命令运行真实 CPU 基线：

```bash
bash code/chapter00_quickstart/run_real_inference.sh
```

脚本默认生成 8 个 token，并设置 `-ngl 0`。可用带空格的第一个参数替换 prompt，用第二个参数替换生成长度；模型和 llama.cpp 不在默认位置时使用脚本帮助中列出的环境变量。默认只显示生成文本和关键耗时，设置 `CHAPTER00_VERBOSE=1` 可查看完整 llama.cpp 日志。

## 扩展入口

| 方向 | 后续入口 |
| --- | --- |
| 模型加载与分词 | [Chapter01](../../source/chapter01_tensor_loader/intro.md) |
| 自研 CPU 执行主线 | [Chapter02–06 教程资产](../../source/reference/tutorial_assets.md) |
| 可选 CUDA 环境 | [CUDA 环境指南](../../source/guides/setup_cuda.md) |
| 其他模型与架构 | [模型支持状态](../../source/reference/model_support.md) |
