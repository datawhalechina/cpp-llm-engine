# 00.2 构建 `llama.cpp` 外部基线

Chapter00 使用固定版本的 `llama.cpp` 产生外部参考结果。CPU 和 CUDA 使用不同构建目录，避免切换配置时删除已经验证的基线。

## 获取已验证版本

从仓库根目录执行：

```bash
LLAMA_CPP_COMMIT=fcc891545b0f06de346d8f67d1e6c61f9bf0e777
test -d llama.cpp/.git || git clone https://github.com/ggml-org/llama.cpp.git llama.cpp
git -C llama.cpp checkout "$LLAMA_CPP_COMMIT"
```

该 commit 与本章脚本使用的 `llama-simple` 参数和输出格式一致。学习时先使用这个版本，避免外部命令变化干扰排错。

## 构建 CPU 基线

```bash
cmake -S llama.cpp -B llama.cpp/build-cpu \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=OFF
cmake --build llama.cpp/build-cpu -j"$(nproc)"
test -x llama.cpp/build-cpu/bin/llama-simple
```

可选 CUDA 构建使用另一目录，不覆盖 CPU 产物：

```bash
cmake -S llama.cpp -B llama.cpp/build-cuda \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=ON
cmake --build llama.cpp/build-cuda -j"$(nproc)"
test -x llama.cpp/build-cuda/bin/llama-simple
```

Chapter00 只要求 CPU 构建成功。CUDA 配置失败时保留 `build-cpu`，继续准备 [`TinyLlama`](./03_download_tinyllama.md)。

## 验证与衔接

| 检查项 | 命令 | 通过条件 |
| --- | --- | --- |
| 源码版本 | `git -C llama.cpp rev-parse HEAD` | 输出已验证 commit |
| CPU 可执行文件 | `test -x llama.cpp/build-cpu/bin/llama-simple` | 退出码为 0 |
| CPU/CUDA 隔离 | `ls -d llama.cpp/build-*` | 两套配置位于不同目录 |
