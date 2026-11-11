# chapter00_quickstart 代码导航

本章用仓库内最小程序检查工具链，再用外部 `llama.cpp` 和 TinyLlama 建立真实推理基线；学习顺序见 [Chapter00 导读](../chapter00_quickstart/intro.md)。

## 代码架构

```text
run.sh -> chapter00_quickstart --check -> 连通性检查通过
run_real_inference.sh -> llama-simple + GGUF -> generated tokens
```

第一条路径不读取模型，第二条路径才证明真实 Tokenizer、权重和生成链已经连通。

## 阅读顺序与资产

| 顺序 | 代码入口 | 观察重点 |
| --- | --- | --- |
| 1 | <a href="https://github.com/datawhalechina/cpp-llm-engine/blob/main/code/chapter00_quickstart/run.sh"><code>run.sh</code></a> | 临时构建目录和 CTest 入口 |
| 2 | <a href="https://github.com/datawhalechina/cpp-llm-engine/blob/main/code/chapter00_quickstart/examples/main_quick.cpp"><code>main_quick.cpp</code></a> | 环境检查与可选模型路径检查 |
| 3 | <a href="https://github.com/datawhalechina/cpp-llm-engine/blob/main/code/chapter00_quickstart/run_real_inference.sh"><code>run_real_inference.sh</code></a> | llama.cpp、GGUF、推理参数和生成结果检查 |
| 4 | <a href="https://github.com/datawhalechina/cpp-llm-engine/blob/main/code/chapter00_quickstart/tests/run_real_inference_contract.sh"><code>run_real_inference_contract.sh</code></a> | 不依赖模型的脚本成功与失败分支 |
| 5 | <a href="https://github.com/datawhalechina/cpp-llm-engine/blob/main/code/chapter00_quickstart/CMakeLists.txt"><code>CMakeLists.txt</code></a> | C++17 target 与测试登记 |

## 运行与验证

先运行无模型检查与脚本契约测试：

```bash
bash code/chapter00_quickstart/run.sh
```

完成 llama.cpp 构建和 TinyLlama 下载后，再运行真实模型：

```bash
bash code/chapter00_quickstart/run_real_inference.sh
```

## 扩展入口

模型加载从 [Chapter01](../chapter01_tensor_loader/intro.md) 继续；CUDA 只作为可选外部基线，环境准备见 [CUDA 环境指南](../guides/setup_cuda.md)。
