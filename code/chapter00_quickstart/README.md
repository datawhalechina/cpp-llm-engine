# Chapter 00：Quickstart 真实推理起点

本目录验证外部真实推理基线，不实现教程自己的 Tensor、Decoder 或 Runtime。完整学习说明见 [`source/chapter00_quickstart`](../../source/chapter00_quickstart/intro.md)。

## 本章资产总览

| 资产 | 作用 |
| --- | --- |
| `CMakeLists.txt` | 构建本章 smoke executable |
| `main_quick.cpp` | 检查模型路径和 quickstart 入口 |
| `run.sh` | 从临时目录配置、构建并运行 smoke |
| `run_real_inference.sh` | 检查本地 llama.cpp 与 GGUF，并执行一次真实推理 |
| `llama.cpp/` | 外部真实推理参考，不属于本章源码资产 |
| `llama.cpp/models/*.gguf` | TinyLlama 等真实模型文件，不提交到仓库 |

## 代码职责

`run.sh` 只验证仓库工具链；`run_real_inference.sh` 只编排已经构建的 llama.cpp 和已经下载的 GGUF。源码克隆、编译和模型下载由学习者按正文显式完成，脚本不会联网或修改外部仓库。

## 运行与验证

从仓库根目录执行：

```bash
bash code/chapter00_quickstart/run.sh
```

该命令只验证本仓库的 smoke executable。准备好 llama.cpp 和 TinyLlama 后，一条命令运行真实 CPU 基线：

```bash
bash code/chapter00_quickstart/run_real_inference.sh
```

脚本默认生成 8 个 token，并设置 `-ngl 0`。可用带空格的第一个参数替换 prompt，用第二个参数替换生成长度；模型和 llama.cpp 不在默认位置时使用脚本帮助中列出的环境变量。默认只显示生成文本和关键耗时，设置 `CHAPTER00_VERBOSE=1` 可查看完整 llama.cpp 日志。
