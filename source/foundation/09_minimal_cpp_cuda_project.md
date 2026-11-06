# F09 构建最小 C++/CUDA 工程

F09 把前面的 C++、CMake、测试和资源管理组合成一个可增长的工程。CPU 项目先跑通，CUDA 作为同一构建图中的可选 target 扩展。

## 从单文件到可测试工程

| 阶段 | 工程结构 | 结果 |
| --- | --- | --- |
| 单文件 | `main.cpp` + executable | 第一次构建成功 |
| 可复用逻辑 | `include/` + `src/` + library | app 和 test 共享实现 |
| 完整工程 | library + app + test + CTest | 可独立验证和持续扩展 |
| CUDA 扩展 | `.cu` target + `ENABLE_CUDA` | CPU baseline 与 GPU backend 并存 |

## 最小构建图

```text
foundation_engine
       ├──> engine_demo
       └──> engine_test
```

| 目录 | 放什么 | 不放什么 |
| --- | --- | --- |
| `include/` | 对外接口 | 大段实现 |
| `src/` | library 实现和资源管理 | app 入口 |
| `app/` | 可运行示例 | 可复用核心逻辑 |
| `tests/` | 固定输入和断言 | 外部大模型依赖 |

## 最小示例

```cmake
add_library(foundation_engine src/engine.cpp)
target_include_directories(foundation_engine PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

add_executable(engine_demo app/main.cpp)
target_link_libraries(engine_demo PRIVATE foundation_engine)

add_executable(engine_test tests/engine_test.cpp)
target_link_libraries(engine_test PRIVATE foundation_engine)
enable_testing()
add_test(NAME engine_test COMMAND engine_test)
```

`Workspace` 等资源对象由 library 持有并自动释放；app 只调用接口，test 只验证行为。CUDA target 以后再挂到同一个接口上，不另起一套工程。

## 验证与练习

```bash
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure
```

对应练习见 [Foundation 正文与练习映射](./exercise_map.md)：FQ04 验证 CMake/CTest，FQ12 验证 workspace 所有权，FQ17 验证三者组合。完成后进入 Chapter 00 和 Chapter 01。
