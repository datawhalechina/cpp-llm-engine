# FQ04 CMake + CTest

## 任务

把一个小工程拆成 library、demo 和 test 三个 target，并用 CTest 注册测试。

## 完成标准

- `cmake -S . -B build` 能完成配置。
- library 被 demo 和 test 正确链接。
- `ctest --test-dir build --output-on-failure` 能发现并运行测试。
- 编译选项挂在 target 上，而不是污染全局环境。

## 验证

复制 starter 并完成 `CMakeLists.txt`，先验证自己的工程：

```bash
cp -r exercises/foundation/problems/fq04_cmake_ctest/starter /tmp/fq04-answer
cmake -S /tmp/fq04-answer -B /tmp/fq04-answer/build
cmake --build /tmp/fq04-answer/build
ctest --test-dir /tmp/fq04-answer/build --output-on-failure
/tmp/fq04-answer/build/fq04_demo
```

再运行题库参考回归：

```bash
bash exercises/foundation/run_all.sh
```

## 提示

- L0：先列出 target 和依赖方向。
- L1：使用 `add_library`、`add_executable`、`target_link_libraries`。
- L2：增加一个失败测试，确认 CTest 确实执行了测试程序。

## 主线映射

对应 [`foundation/02_cmake_build_basics`](/foundation/02_cmake_build_basics)。
