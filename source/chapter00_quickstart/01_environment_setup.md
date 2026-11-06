# 00.1 检查本地环境

完成本节后，终端应能找到 C++ 编译器、CMake 和 Git。CUDA 不属于 Chapter00 CPU 基线的前置条件；需要 GPU 时再阅读 [`CUDA 环境指南`](../guides/setup_cuda.md)。

## Linux 与 Windows 选择

| 平台 | 推荐方式 | 额外要求 | 当前支持状态 |
| --- | --- | --- | --- |
| Linux | Ubuntu 22.04 或兼容发行版 | Bash、GCC/Clang、CMake、Git | 主要开发与 CI 路径 |
| Windows + WSL2 | WSL2 中安装 Ubuntu，在 Linux 文件系统内克隆仓库 | 开启虚拟化；所有工具都安装在 WSL 内 | Windows 推荐路径，复用 Linux 命令 |
| Windows 原生 | Visual Studio C++、CMake、Git 符号链接和 PowerShell 脚本适配 | 需要替换 Bash、`/tmp` 和 Linux 路径假设 | 尚未纳入 CI，不作为当前发布契约 |

Windows 上首次安装 WSL2 时，在管理员 PowerShell 中执行：

```powershell
wsl --install -d Ubuntu-22.04
```

进入 WSL 后，在 `~/` 下克隆仓库，不要优先放到 `/mnt/c/`；Linux 文件系统在大量小文件编译、权限和符号链接方面更稳定。不要混用 Windows 版 CMake/Node 和 WSL 内的编译器。

## 最小环境

| 项目 | 建议基线 | 用途 |
| --- | --- | --- |
| 操作系统 | Ubuntu 22.04 或兼容 Linux | 运行仓库脚本和 llama.cpp |
| CMake | 3.20+ | 配置 C++ 项目 |
| 编译器 | GCC 11+ 或 Clang 14+ | 编译 C++17 代码 |
| Git | 可用版本 | 获取并固定 llama.cpp 源码 |
| 磁盘 | 至少 3 GiB 可用空间 | 保存源码、构建产物和 638 MiB GGUF |
| 内存 | 建议至少 4 GiB 可用内存 | 加载 TinyLlama Q4_K_M 并执行 CPU 推理 |

C++17 是项目构建契约，不是可选建议。各模块 CMake target 都要求 `cxx_std_17`，代码使用了 `std::string_view`、`std::optional`、`std::variant`、`std::filesystem` 和 `std::byte` 等 C++17 能力。C++11 或 C++14 可以用来学习部分 Foundation 语法题，但不能构建完整仓库。

在 Ubuntu 22.04 或兼容 Debian 系统上，可先安装 CPU 主线工具：

```bash
sudo apt update
sudo apt install -y build-essential cmake git curl
```

Chapter00 本身不依赖 Node.js。如果要运行根目录的 `npm run test:regression`、文档检查或 VitePress，还需要 Node.js 20+ 和 npm；安装后在仓库根目录执行 `npm install`。

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

需要全仓库回归时，再检查 `node --version` 和 `npm --version`。需要 GPU 时，独立按 [`CUDA 环境指南`](../guides/setup_cuda.md) 检查驱动、Toolkit 和设备，不要把 CUDA 问题混入 Chapter00 CPU 基线。

WSL2 使用 GPU 时，Windows 宿主机安装支持 WSL 的 NVIDIA 驱动，CUDA Toolkit 和教程构建工具安装在 WSL 内。先确认 WSL 内 `nvidia-smi` 可见设备，再继续 CUDA 章节。
