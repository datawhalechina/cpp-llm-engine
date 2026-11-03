# F02 CMake 与构建基础

CMake 在本教程中负责把源码、library、app 和 test 组织成可重复构建的 target。先掌握 target 依赖，再进入 Chapter 00 和后续章节。

## 主线位置与学习目标

| 需要掌握 | 直接用途 |
| --- | --- |
| `add_library`、`add_executable` | 拆分 engine、demo 和测试 |
| `target_link_libraries` | 表达依赖方向 |
| `target_include_directories` | 暴露最小接口 |
| out-of-source build | 隔离源码和产物 |
| Debug / Release | 分开做正确性和性能验证 |

## 最小构建图

```text
core library
   ├──> app
   └──> test
```

| target | 输入 | 输出 | 责任 |
| --- | --- | --- | --- |
| `core` | `src/core.cpp`、公开头文件 | library | 可复用逻辑 |
| `app` | `main.cpp` + `core` | executable | 展示运行结果 |
| `test` | 测试代码 + `core` | executable / CTest | 固定行为 |

## 最小示例

```cmake
cmake_minimum_required(VERSION 3.20)
project(mini_engine LANGUAGES CXX)

add_library(core src/core.cpp)
target_compile_features(core PUBLIC cxx_std_17)
target_include_directories(core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

add_executable(app app/main.cpp)
target_link_libraries(app PRIVATE core)

add_executable(core_test tests/core_test.cpp)
target_link_libraries(core_test PRIVATE core)
enable_testing()
add_test(NAME core_test COMMAND core_test)
```

编译选项优先挂在 target 上，不用全局变量污染其他模块。

## 验证与练习

```bash
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure
```

对应练习见 [Foundation 正文与练习映射](./exercise_map.md)，直接完成 `FQ04`；把 library、app、test 和 workspace 组合起来时完成 `FQ17`。
