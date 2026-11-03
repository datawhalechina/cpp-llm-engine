# Foundation 正文与练习映射

下表连接 Foundation 正文、对应题目、验证目标和主线出口。题目要求、starter、编译命令和验收标准见 [`Foundation Exercises`](../exercises/foundation/README.md)。

| 正文 | 题目 | 关系 | 题目验证什么 | 主线出口 |
| --- | --- | --- | --- | --- |
| [F01 C++](./01_cxx_basics.md) | [`FQ01`](../exercises/foundation/problems/fq01_move_buffer/README.md) | 直接练习 | move-only 所有权和资源释放 | Foundation 工程基础 |
| [F02 CMake](./02_cmake_build_basics.md) | [`FQ04`](../exercises/foundation/problems/fq04_cmake_ctest/README.md) | 直接练习 | target、构建和 CTest 注册 | Foundation 工程基础 |
| [F03 CUDA 环境](./03_cuda_environment_basics.md) | [`FQ10`](../exercises/foundation/problems/fq10_vector_add/README.md) | 主线练习 | CPU reference、kernel 索引和边界 | Chapter 03 CUDA backend |
| [F04 调试测试](./04_debugging_testing_basics.md) | [`FQ05`](../exercises/foundation/problems/fq05_debug_safe_buffer/README.md) | 直接练习 | 输入检查、失败路径和回归用例 | Foundation 验证方法 |
| [F05 对象设计](./05_oop_and_design_basics.md) | [`FQ01`](../exercises/foundation/problems/fq01_move_buffer/README.md)、[`FQ08`](../exercises/foundation/problems/fq08_mini_tensor/README.md) | 直接练习 | 资源所有权、对象组合和 Tensor 接口 | Chapter 01 Tensor |
| [F06 数据结构](./06_data_structure_algorithm_basics.md) | [`FQ03`](../exercises/foundation/problems/fq03_tensor_view/README.md)、[`FQ06`](../exercises/foundation/problems/fq06_cpu_matmul/README.md) | 直接练习 | 数据布局、view 和 CPU reference | Chapter 02-03 |
| [F07 CUDA 库](./07_cuda_common_libraries.md) | [`FQ10`](../exercises/foundation/problems/fq10_vector_add/README.md) | 主线练习 | backend 映射和 CPU/CUDA 对照 | Chapter 03 CUDA backend |
| [F08 PyTorch/C++](./08_pytorch_to_cpp_bridge.md) | [`FQ08`](../exercises/foundation/problems/fq08_mini_tensor/README.md) | 直接练习 | Tensor ownership、view 和索引语义 | Chapter 01 Tensor |
| [F09 最小项目](./09_minimal_cpp_cuda_project.md) | [`FQ04`](../exercises/foundation/problems/fq04_cmake_ctest/README.md)、[`FQ12`](../exercises/foundation/problems/fq12_raii_workspace/README.md)、[`FQ17`](../exercises/foundation/problems/fq17_minimal_project/README.md) | 综合练习 | 工程构建、library/app/test 组织和资源边界 | Chapter 00-01 |

F03 的环境排查通过 CUDA probe 验证；F07 的 `cuBLAS` 通过 `code/cuda` 验证，CUDA Graph、CUB 和 NCCL 作为原理阅读；F08 的 `state_dict -> ModelLoader` 由 Chapter01 的真实 Loader 验证。
