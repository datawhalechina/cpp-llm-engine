# Foundation Exercises 基础小题库

这套题库把 `foundation` 的最低门槛落实为可编译、可测试的小题。题目默认不需要模型文件或 GPU；FQ10 提供 CPU 参考实现和可选 CUDA starter。

## 当前题目

| 编号 | 难度 | 主题 | 验证重点 | 对应正文 / 主线 |
| --- | --- | --- | --- | --- |
| `FQ01` | 入门 | move-only Buffer | 所有权、移动、释放 | `F01` / `foundation` |
| `FQ02` | 入门 | Tensor shape | shape、numel、边界 | `chapter01` |
| `FQ03` | 主线 | Tensor view | offset、共享存储 | `F06` / `chapter02` |
| `FQ04` | 入门 | CMake + CTest | target、构建、测试注册 | `F02` / `foundation` |
| `FQ05` | 入门 | 安全缓冲区 | 输入检查、失败路径 | `F04` / `foundation` |
| `FQ06` | 主线 | CPU MatMul | shape 检查、数值结果 | `F06` / `chapter03` |
| `FQ07` | 主线 | Token queue | FIFO、状态转移 | `F06` / `chapter06` |
| `FQ08` | 主线 | Mini Tensor | ownership、view、索引组合 | `F05`、`F08` / `chapter01` |
| `FQ09` | 主线 | Request State | 状态迁移、终态、取消 | `chapter06` |
| `FQ10` | 进阶选修 | Vector Add / CUDA 映射 | CPU 参考、kernel 索引、边界 | `F03`、`F07` / `chapter03` |
| `FQ11` | 主线 | Request Queue | 容量、FIFO、取消、backpressure | `chapter06` |
| `FQ12` | 主线 | RAII Workspace | 移动所有权、边界、异常安全 | `F09` / `chapter06` / `reference` |
| `FQ13` | 主线 | Shape-aware Batch | sequence length、padding、buffer offset | `chapter06` |
| `FQ14` | 主线 | Step Scheduler | FIFO、token budget、round-robin | `chapter06` |
| `FQ15` | 主线 | KV Block Pool | block 分配、释放、复用 | `chapter06` / `reference` |
| `FQ16` | 综合 | Prefill/Decode State | 阶段迁移、token 计数、终态 | `chapter06` |
| `FQ17` | 综合 | Minimal Project Integration | library、app、test、workspace 所有权 | `F09` / `chapter00` / `chapter01` |

每道题都包含问题说明、完成标准、验证命令、提示和主线链接。题目入口：

- [`FQ01 move-only Buffer`](./problems/fq01_move_buffer/README.md)
- [`FQ02 Tensor shape`](./problems/fq02_tensor_shape/README.md)
- [`FQ03 Tensor view`](./problems/fq03_tensor_view/README.md)
- [`FQ04 CMake + CTest`](./problems/fq04_cmake_ctest/README.md)
- [`FQ05 安全缓冲区`](./problems/fq05_debug_safe_buffer/README.md)
- [`FQ06 CPU MatMul`](./problems/fq06_cpu_matmul/README.md)
- [`FQ07 Token queue`](./problems/fq07_token_queue/README.md)
- [`FQ08 Mini Tensor`](./problems/fq08_mini_tensor/README.md)
- [`FQ09 Request State`](./problems/fq09_request_state/README.md)
- [`FQ10 Vector Add / CUDA Mapping`](./problems/fq10_vector_add/README.md)
- [`FQ11 Request Queue`](./problems/fq11_request_queue/README.md)
- [`FQ12 RAII Workspace`](./problems/fq12_raii_workspace/README.md)
- [`FQ13 Shape-aware Batch`](./problems/fq13_shape_aware_batch/README.md)
- [`FQ14 Step Scheduler`](./problems/fq14_step_scheduler/README.md)
- [`FQ15 KV Block Pool`](./problems/fq15_kv_block_pool/README.md)
- [`FQ16 Prefill/Decode State`](./problems/fq16_prefill_decode_state/README.md)
- [`FQ17 Minimal Project Integration`](./problems/fq17_minimal_project/README.md)

当前目录中的 `.cpp` 是可运行参考实现；每道题的 `starter/` 目录提供不含答案的最小起点。

`bash scripts/check_foundation_starters.sh` 会编译所有 C++ starter，并检查 FQ04 的 CMake starter 与 FQ10 的 CUDA starter 文件存在。它验证 starter 能作为起点构建，不替代题目本身的行为测试。

`bash scripts/check_foundation_metadata.sh` 会检查每道题的 README、starter 和统一栏目是否存在。

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
| 主线出口 | Chapter 01-05 的 CPU 推理引擎 | Chapter 03 CUDA backend、后续 GPU kernel 和 benchmark |
| 当前完成度 | 主线完整，可作为默认学习路径 | 入门映射已具备，尚未覆盖完整 CUDA allocator、stream、event、kernel benchmark |

当前 Foundation 的 GPU 资产是 CPU 主线的选修入口，不是独立完整 GPU 题库。后续扩展 GPU 题库时，应优先补 `device memory -> stream/event -> kernel benchmark`，并复用 CPU 题库已经固定的输入、输出和正确性标准。

## 两道状态题的区别

- `FQ09 Request State` 关注请求生命周期：admission、prefill 完成、生成、完成和取消。
- `FQ16 Prefill/Decode State` 关注模型执行阶段：一次 prompt prefill 如何切换到逐 token decode，以及生成计数如何触发终态。

先做 `FQ09`，再做 `FQ14` 和 `FQ15`，最后做 `FQ16`，可以把请求状态、调度动作、KV 资源和模型执行阶段分开理解。

## Foundation 完成标准

完成入门组后，学习者应能独立构建和测试 C++ 小程序；完成 CPU 主线和 FQ17 后，应能读懂 Chapter 00-03 的项目组织、对象和算子；完成 Runtime 综合练习后，应能进入 Chapter 06 阅读 Request、Scheduler、KV Cache 和 Prefill/Decode。题库参考实现的 CTest 通过不等于学习完成，学习者还需要把 starter 改完并运行对应验证。

## 运行全部题目

```bash
bash exercises/foundation/run_all.sh
```

也可以手动构建：

```bash
cmake -S exercises/foundation -B exercises/foundation/build
cmake --build exercises/foundation/build
ctest --test-dir exercises/foundation/build --output-on-failure
npm run test:regression
```

内存安全题可以额外启用 Sanitizer：

```bash
cmake -S exercises/foundation -B exercises/foundation/build-asan -DENABLE_ASAN=ON
cmake --build exercises/foundation/build-asan
ctest --test-dir exercises/foundation/build-asan --output-on-failure
```

当前受控执行环境会限制 AddressSanitizer 的运行时和 LeakSanitizer；如果出现 `DEADLYSIGNAL` 或 `LeakSanitizer has encountered a fatal error`，应在普通本地环境重跑。当前环境使用 UBSan 做可运行验证：

```bash
cmake -S exercises/foundation -B exercises/foundation/build-ubsan \
  -DCMAKE_CXX_FLAGS=-fsanitize=undefined
cmake --build exercises/foundation/build-ubsan
ctest --test-dir exercises/foundation/build-ubsan --output-on-failure
```

## 题目规范

学习目标统一写成“动作 + 对象 + 条件 + 验证结果”。例如：完成 `FQ03` 后，应能在不复制底层数组的前提下创建 view，并通过测试确认修改 view 会反映到原始 storage。
