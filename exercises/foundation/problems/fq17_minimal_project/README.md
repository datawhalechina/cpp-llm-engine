# FQ17 Minimal Project Integration

## 任务

把一个最小的推理 engine step 组织成可独立构建的 C++ 工程：library 提供 `EngineStep` 和 move-only `Workspace`，app 调用一次 step，test 验证结果和资源边界。

## 完成标准

- `cmake -S . -B build` 能配置，`cmake --build build` 能构建 `foundation_engine`、`engine_demo` 和 `engine_test`。
- `engine_demo` 能调用 library 并输出 `result: 42`。
- `ctest --test-dir build --output-on-failure` 能运行测试。
- `Workspace` 禁止复制，支持移动，并在离开作用域时自动释放资源。
- 编译选项挂在 library、app、test 对应 target 上，不依赖全局目录状态。

## 验证

在临时目录完成 starter 后运行：

```bash
cp -r exercises/foundation/problems/fq17_minimal_project/starter /tmp/fq17-answer
cmake -S /tmp/fq17-answer -B /tmp/fq17-answer/build
cmake --build /tmp/fq17-answer/build
ctest --test-dir /tmp/fq17-answer/build --output-on-failure
/tmp/fq17-answer/build/engine_demo
```

预期输出：

```text
result: 42
```

参考实现运行：

```bash
bash exercises/foundation/run_all.sh
```

## 提示

- L0：先画出 `foundation_engine -> engine_demo` 和 `foundation_engine -> engine_test` 的依赖图。
- L1：将 `Workspace` 的声明放入 `include/engine.h`，实现放入 `src/engine.cpp`。
- L2：使用 `std::unique_ptr<int[]>`，删除复制操作，给 library、app 和 test 分别设置 target 属性。

## 选做：Backend 与 Factory

基础验收通过后，把 `EngineStep` 的计算实现改造成可替换后端：

1. 定义只有 `virtual int run(int input) = 0` 的 `Backend` 接口。
2. 用 `CpuBackend` 实现当前的 `input + 1` 行为。
3. 增加 `make_backend("cpu")`，返回 `std::unique_ptr<Backend>`；未知名称应抛出明确异常。
4. 让 `EngineStep` 通过构造函数接收 `std::unique_ptr<Backend>`，不在内部判断后端名称。
5. 在测试中注入返回固定值的 `FakeBackend`，证明 `EngineStep` 依赖接口而不是 CPU 实现。

这个扩展不改变基础题的 `result: 42` 验收。它分别练习三项职责：Backend 接口定义可替换边界，Factory 集中创建实现，Fake Backend 隔离测试。对应真实代码中的 CPU/CUDA 执行后端和请求执行器工厂。

## 主线映射

对应 [`foundation/09_minimal_cpp_cuda_project`](/foundation/09_minimal_cpp_cuda_project)、[`foundation/02_cmake_build_basics`](/foundation/02_cmake_build_basics)、[`foundation/05_oop_and_design_basics`](/foundation/05_oop_and_design_basics) 和 [`chapter01_tensor_loader`](/chapter01_tensor_loader/intro)。
