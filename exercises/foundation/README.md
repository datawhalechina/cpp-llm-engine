# Foundation Exercises 基础小题库

这套题库把 `foundation` 的最低门槛落实为可编译、可测试的小题。题目默认不需要模型文件或 GPU；FQ10 提供 CPU 参考实现和可选 CUDA starter。

## 题目资产

| 编号 | 难度 | 主题 | 验证重点 | 对应正文 / 主线 |
| --- | --- | --- | --- | --- |
| [`FQ01`](./problems/fq01_move_buffer/README.md) | 入门 | 仅移动缓冲区 | 所有权、移动、释放 | `F01` / Foundation |
| [`FQ02`](./problems/fq02_tensor_shape/README.md) | 入门 | Tensor shape | shape、numel、边界 | `F06` / Chapter01 |
| [`FQ03`](./problems/fq03_tensor_view/README.md) | 主线 | Tensor view | offset、共享存储 | `F06` / Chapter02 |
| [`FQ04`](./problems/fq04_cmake_ctest/README.md) | 入门 | CMake + CTest | target、构建、测试注册 | `F02` / Foundation |
| [`FQ05`](./problems/fq05_debug_safe_buffer/README.md) | 入门 | 安全缓冲区 | 输入检查、失败路径 | `F04` / Foundation |
| [`FQ06`](./problems/fq06_cpu_matmul/README.md) | 主线 | CPU MatMul | shape 检查、数值结果 | `F06` / Chapter03 |
| [`FQ07`](./problems/fq07_token_queue/README.md) | 主线 | 词元队列 | FIFO、状态转移 | `F06`、`F10` / Chapter08 |
| [`FQ08`](./problems/fq08_mini_tensor/README.md) | 主线 | 最小 Tensor | 所有权、视图、索引组合 | `F05`、`F08` / Chapter01–02 |
| [`FQ09`](./problems/fq09_request_state/README.md) | 主线 | Request State | 状态迁移、终态、取消 | `F10` / Chapter06、08 |
| [`FQ10`](./problems/fq10_vector_add/README.md) | 进阶选修 | 向量加法与 CUDA 映射 | CPU 参考、核函数索引、边界 | `F03`、`F07` / Chapter03 |
| [`FQ11`](./problems/fq11_request_queue/README.md) | 主线 | 请求队列 | 容量、FIFO、取消、背压 | `F10` / Chapter08 |
| [`FQ12`](./problems/fq12_raii_workspace/README.md) | 主线 | RAII Workspace | 移动所有权、边界、异常安全 | `F01`、`F09` / Chapter06 |
| [`FQ13`](./problems/fq13_shape_aware_batch/README.md) | 主线 | 感知形状的批次 | 序列长度、填充、缓冲区偏移 | `F06` / Chapter08 |
| [`FQ14`](./problems/fq14_step_scheduler/README.md) | 主线 | 单步调度器 | FIFO、词元预算、轮转推进 | `F10` / Chapter08 |
| [`FQ15`](./problems/fq15_kv_block_pool/README.md) | 主线 | KV Block Pool | block 分配、释放、复用 | `F06` / Chapter07 |
| [`FQ16`](./problems/fq16_prefill_decode_state/README.md) | 综合 | Prefill/Decode State | 阶段迁移、token 计数、终态 | `F10` / Chapter06 |
| [`FQ17`](./problems/fq17_minimal_project/README.md) | 综合 | Minimal Project Integration | library、app、test、workspace 所有权 | `F02`、`F09` / Chapter00–01 |

每道题都包含问题说明、完成标准、答案验证命令、提示和主线链接。验证顺序固定为：复制并完成 starter，先运行学习者答案，再运行 `run_all.sh` 检查参考实现没有回归。

题库根目录中的 `.cpp` 是可运行参考实现；每道题的 `starter/` 目录提供不含答案的最小起点。单文件答案统一使用 `bash scripts/check_foundation_answer.sh /tmp/<answer>/main.cpp` 编译并运行；FQ04 和 FQ17 按各自 README 使用 CMake/CTest 验收。

## 如何完成一道题

以 `FQ06` 为例，先阅读题目说明，再复制 starter 到临时目录，完成后独立编译：

```bash
cp -r exercises/foundation/problems/fq06_cpu_matmul/starter /tmp/fq06-answer
c++ -std=c++17 -Wall -Wextra -Wpedantic \
  /tmp/fq06-answer/main.cpp -o /tmp/fq06-answer/fq06
/tmp/fq06-answer/fq06
```

然后运行题库参考实现，确认主线回归仍然通过：

```bash
bash exercises/foundation/run_all.sh
```

`FQ04` 的 starter 是 `CMakeLists.txt` 设计说明，需要在临时工程中完成 target 和 CTest；`FQ10` 先编译 `starter/main.cpp` 的 CPU 版本，有 CUDA 环境再单独编译 `starter/main.cu`。不要直接修改题库根目录的参考实现。

## 推荐题库顺序

不要按编号一次性刷完，按目标选择路径：

| 路径 | 题目 | 完成后能做什么 |
| --- | --- | --- |
| 入门必做 | `FQ01`、`FQ02`、`FQ04`、`FQ05` | 构建小型 C++ 程序，处理所有权、shape 和错误输入 |
| CPU 主线 | `FQ03`、`FQ06`、`FQ08`、`FQ12` | 理解 view、CPU 算子、Tensor ownership 和 workspace |
| Runtime 综合练习 | `FQ07`、`FQ09`、`FQ11`、`FQ13`、`FQ14`、`FQ15`、`FQ16` | 理解 Request、batch、scheduler、KV block 和 Prefill/Decode |
| 工程整合 | `FQ17` | 把 library、app、test 和资源所有权组合成可运行项目 |
| CUDA 选修 | `FQ10` | 先写 CPU reference，再把索引逻辑映射到 CUDA kernel |

推荐总顺序：`FQ01 -> FQ02 -> FQ04 -> FQ05 -> FQ03 -> FQ06 -> FQ08 -> FQ12 -> FQ17 -> FQ07 -> FQ09 -> FQ11 -> FQ13 -> FQ14 -> FQ15 -> FQ16`，有 CUDA 环境再插入 `FQ10`。

初学者完成入门组即可进入 `chapter00`；希望理解 generation runtime 的学习者继续完成主线组；已有 GPU 基础的学习者再做 FQ10。

## CPU / GPU 题库资产对照

| 资产维度 | CPU 题库 | GPU 题库 |
| --- | --- | --- |
| 题目范围 | FQ01-FQ09、FQ11-FQ17 | FQ10，以及 F03/F07 文档练习 |
| 核心对象 | ownership、Tensor、view、MatMul、Request、batch、scheduler、KV block、Prefill/Decode | device、thread index、kernel 边界、host/device 数据搬运、CUDA 库 |
| 默认环境 | C++17、CMake、CTest，无模型、无 GPU | CPU 可完成 reference；CUDA kernel 需要 CUDA toolkit 和 GPU |
| 验证方式 | 17 题 CTest、starter 编译、元数据检查、回归基线 | FQ10 CPU reference 默认验证；CUDA starter 需要独立 CUDA 工程对照验证 |
| 主线出口 | Chapter 01-08 的数据、算子、Runtime、缓存与调度 | Chapter 03 CUDA backend、后续 GPU kernel 和 benchmark |

Foundation 的 GPU 练习先用 FQ10 建立 CPU/CUDA 对照；继续学习设备内存、stream、event、更多 kernel 和 benchmark 时，进入 [`code/cuda`](/code/cuda)。

## 两道状态题的区别

- `FQ09 Request State` 关注请求生命周期：admission、prefill 完成、生成、完成和取消。
- `FQ16 Prefill/Decode State` 关注模型执行阶段：一次 prompt prefill 如何切换到逐 token decode，以及生成计数如何触发终态。

先做 `FQ09`，再做 `FQ14` 和 `FQ15`，最后做 `FQ16`，可以把请求状态、调度动作、KV 资源和模型执行阶段分开理解。

## Foundation 完成标准

完成入门组后，应能独立构建和测试 C++ 小程序；完成 CPU 主线和 FQ17 后，应能读懂 Chapter00–03 的项目组织、对象和算子；完成 Runtime 综合练习后，应能分别阅读 Chapter06 的生成状态、Chapter07 的 K/V 资源和 Chapter08 的调度。参考实现 CTest 通过只说明题库基线有效，自己的 starter 还需通过对应答案命令。

## 运行全部题目

```bash
bash exercises/foundation/run_all.sh
```

也可以手动构建：

```bash
cmake -S exercises/foundation -B exercises/foundation/build
cmake --build exercises/foundation/build
ctest --test-dir exercises/foundation/build --output-on-failure
```

内存安全题可以额外启用 Sanitizer：

```bash
cmake -S exercises/foundation -B exercises/foundation/build-asan -DENABLE_ASAN=ON
cmake --build exercises/foundation/build-asan
ctest --test-dir exercises/foundation/build-asan --output-on-failure
```

容器或受限环境可能无法运行 AddressSanitizer/LeakSanitizer；遇到运行时权限错误时，可先用 UBSan 检查未定义行为，再在本地 Linux 环境重跑 ASan：

```bash
cmake -S exercises/foundation -B exercises/foundation/build-ubsan \
  -DCMAKE_CXX_FLAGS=-fsanitize=undefined
cmake --build exercises/foundation/build-ubsan
ctest --test-dir exercises/foundation/build-ubsan --output-on-failure
```
