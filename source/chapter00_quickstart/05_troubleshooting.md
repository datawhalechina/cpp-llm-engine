# 00.5 排查首次运行问题

排错顺序固定为“工具 -> 源码 -> 可执行文件 -> 模型 -> 运行参数 -> GPU”。先恢复 CPU 基线，再判断是否需要处理 CUDA。

## 现象与处理

| 现象 | 检查命令 | 处理方式 |
| --- | --- | --- |
| `llama.cpp` 已存在但不是 Git 仓库 | `test -d llama.cpp/.git` | 将误建目录改名后重新 clone；不要移动正常仓库 |
| 找不到 `llama-simple` | `find llama.cpp -path '*/bin/llama-simple'` | 完成 00.2，或设置 `LLAMA_CPP_BUILD_DIR` |
| 找不到 GGUF | `ls -lh llama.cpp/models/*.gguf` | 完成 00.3，或设置 `CPP_LLM_ENGINE_MODEL_PATH` |
| SHA256 不一致 | `sha256sum <model.gguf>` | 删除损坏文件并重新下载 |
| CPU 推理失败 | `CHAPTER00_VERBOSE=1 bash code/chapter00_quickstart/run_real_inference.sh` | 根据完整 llama.cpp 日志检查模型和参数 |
| CUDA 构建失败 | `nvcc --version`、`nvidia-smi` | 保留 `build-cpu`，按 CUDA 环境指南检查驱动和 Toolkit |
| 显存不足 | `nvidia-smi` | 减少 `CHAPTER00_GPU_LAYERS` 或回到 CPU |
| 运行较慢 | 查看 `prompt eval` 和 `eval` 速度 | 先确认能稳定生成，再进入 Benchmark 章节测量 |

## 最小回退路径

以下三条命令把问题收敛到仓库工具链、外部构建或真实模型之一：

```bash
bash code/chapter00_quickstart/run.sh
test -x llama.cpp/build-cpu/bin/llama-simple || test -x llama.cpp/build/bin/llama-simple
bash code/chapter00_quickstart/run_real_inference.sh
```

第一条失败时检查 C++/CMake；第二条失败时重做 llama.cpp 构建；前两条通过而第三条失败时，检查 GGUF 和运行参数。

Chapter00 使用已经成熟的 llama.cpp 可执行文件；Chapter01 开始构建本教程自己的 Loader、Tokenizer 和模型接口。因此 Chapter00 能推理但 Chapter01 测试失败，通常说明自研代码、编译环境或输入校验尚未通过，不能用外部基线成功替代对应章节测试。

## 验证与衔接

| 状态 | 结论 |
| --- | --- |
| 连通性检查失败 | 仓库工具链尚未可用 |
| 连通性检查通过、`llama-simple` 缺失 | 外部基线尚未构建 |
| 可执行文件存在、模型校验失败 | GGUF 下载不完整或路径错误 |
| CPU 真实推理通过 | Chapter00 完成，可以进入 Chapter01 |
