# 教学型推理项目

这些项目分别压缩 C++/CUDA 执行、Python 算法骨架和调度缓存系统。它们用于降低成熟框架源码的阅读门槛，不替代本教程的 C++ 主线。

<a id="tiny-vllm"></a>
## Tiny-VLLM

### 阅读目的

[Tiny-VLLM](https://github.com/jmaczan/tiny-vllm) 用 C++ 和 CUDA 串起权重读取、RMSNorm、RoPE、GEMM、Attention、K/V 和分页管理，适合观察一条设备执行路径怎样逐步增长。

### 重点对象

| 对象 | 官方入口 | 阅读问题 |
| --- | --- | --- |
| 项目实现 | [Tiny-VLLM 仓库](https://github.com/jmaczan/tiny-vllm) | 模型结构怎样映射到 C++ 数据、CUDA 计算核和 cuBLAS 调用 |

### 对应章节

Foundation 对照 CUDA 线程、内存和计算核；Chapter03–06 对照 GEMM、Attention、量化和生成；`code/cuda` 对照单卡设备实现。

<a id="nano-vllm"></a>
## nano-vLLM

### 阅读目的

[nano-vLLM](https://github.com/GeeeekExplorer/nano-vllm) 用紧凑 Python 表达模型执行和 vLLM 风格运行时，适合先理解对象关系与控制流，再回到本教程处理 C++ 所有权、错误契约和后端。

### 重点对象

| 对象 | 官方入口 | 阅读问题 |
| --- | --- | --- |
| 项目实现 | [nano-vLLM 仓库](https://github.com/GeeeekExplorer/nano-vllm) | 模型、调度器、块管理器和执行器怎样传递状态 |

### 对应章节

Chapter04 对照模型层，Chapter06 对照单请求循环，Chapter07–08 对照分页缓存与调度，Chapter09 对照优化观测点。

<a id="mini-sglang"></a>
## Mini-SGLang

### 阅读目的

[Mini-SGLang](https://github.com/sgl-project/mini-sglang) 保留现代推理服务中的预填充、解码、Radix Cache 和重叠调度，适合观察缓存命中怎样进入调度循环。

### 重点对象

| 对象 | 官方入口 | 阅读问题 |
| --- | --- | --- |
| 调度器 | [`scheduler.py`](https://github.com/sgl-project/mini-sglang/blob/main/python/minisgl/scheduler/scheduler.py) | 普通循环和重叠循环怎样交换批次与结果 |
| 项目实现 | [Mini-SGLang 仓库](https://github.com/sgl-project/mini-sglang) | 缓存、调度和模型执行怎样组成最小系统 |

### 对应章节

Chapter06 对照请求阶段，Chapter07 对照前缀复用，Chapter08 对照分块预填充与重叠调度，Chapter09–10 对照性能和服务生命周期。
