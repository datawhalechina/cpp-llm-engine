# FQ12 RAII Workspace

## 任务

实现一个只能移动的 workspace，模拟 Executor 为一次 step 获取的临时缓冲区，并保证离开作用域时自动释放。

## 完成标准

- workspace 独占底层数组，禁止复制。
- 移动构造和移动赋值后，源对象不再拥有资源。
- 空 workspace 和越界访问明确失败。
- 不需要手工 `delete`，异常路径也能安全结束。

## 验证

运行 `bash exercises/foundation/run_all.sh`，重点测试 `fq12_raii_workspace`；再用 UBSan/ASan 检查移动和边界路径。

## 提示

- L0：先确定 workspace 的空状态和所有权状态。
- L1：使用 `std::unique_ptr<float[]>`，显式删除复制操作。
- L2：增加 move assignment、异常退出和多次 step 获取/释放测试。

## 主线映射

对应 [`推理引擎内存管理`](/reference/memory_management)、[`Foundation C++ 基础`](/foundation/01_cxx_basics) 和 [`Chapter06 Generation Core`](/chapter06_generation/next/intro)。
