# 00.1 检查本地环境

完成本节后，终端应能找到 C++ 编译器、CMake 和 Git。CUDA 不属于 Chapter00 CPU 基线的前置条件；需要 GPU 时再阅读 [`CUDA 环境指南`](../guides/setup_cuda.md)。

## 最小环境

| 项目 | 建议基线 | 用途 |
| --- | --- | --- |
| 操作系统 | Ubuntu 22.04 或兼容 Linux | 运行仓库脚本和 llama.cpp |
| CMake | 3.20+ | 配置 C++ 项目 |
| 编译器 | GCC 11+ 或 Clang 14+ | 编译 C++17 代码 |
| Git | 可用版本 | 获取并固定 llama.cpp 源码 |
| 磁盘 | 至少 3 GiB 可用空间 | 保存源码、构建产物和 638 MiB GGUF |
| 内存 | 建议至少 4 GiB 可用内存 | 加载 TinyLlama Q4_K_M 并执行 CPU 推理 |

## 检查命令

从仓库根目录执行：

```bash
cmake --version
g++ --version || clang++ --version
git --version
bash code/chapter00_quickstart/run.sh
```

最后一条命令不读取模型，只编译 Chapter00 连通性检查程序。看到 `smoke check passed` 后，继续构建 [`llama.cpp`](./02_clone_and_build_llamacpp.md)。

## 验证与衔接

| 检查项 | 通过条件 | 失败时先看 |
| --- | --- | --- |
| CMake | 版本不低于 3.20 | PATH 或系统软件源 |
| C++ 编译器 | 能报告版本并支持 C++17 | Foundation C++/CMake 基础 |
| Git | 能报告版本 | PATH 或系统软件源 |
| 仓库连通性检查 | 输出 `smoke check passed` | [`00.5 排查首次运行问题`](./05_troubleshooting.md) |
