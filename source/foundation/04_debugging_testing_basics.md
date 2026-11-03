# F04 调试与测试基础

推理引擎的错误通常落在环境、构建、shape、数值或性能中的一个范围。调试的第一步不是加更多日志，而是用最小输入把失败归类。

## 主线位置与学习目标

| 失败类型 | 先检查 | 常用证据 |
| --- | --- | --- |
| 环境 | driver、toolkit、模型路径 | 命令退出码、设备枚举 |
| 构建 | target、include、链接 | CMake 配置和编译日志 |
| shape | rank、维度、offset | 断言、打印 shape |
| 数值 | 中间 Tensor 和参考结果 | 固定输入、误差阈值 |
| 性能 | 测量方法和输入规模 | benchmark、profile |

## 最小验证循环

```text
最小输入 -> 复现失败 -> 检查 shape -> 检查数值 -> 保留回归测试 -> 再测性能
```

测试类型按范围分开：

| 类型 | 适合验证 | 例子 |
| --- | --- | --- |
| 单元测试 | 一个函数或对象 | `Tensor::reshape` |
| 集成测试 | 多个 target 的接口 | loader + metadata |
| 回归测试 | 已修复的错误 | 错误 shape、取消路径 |
| benchmark | 吞吐和延迟 | CPU MatMul、decode step |

## 最小示例

```cpp
int safe_read(const std::vector<int>& values, std::size_t index) {
  assert(index < values.size());
  return values[index];
}
```

测试至少覆盖一个正常输入和一个失败输入；失败案例保留下来，避免后续重构重新引入同一错误。

## 验证与练习

```bash
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure
```

对应练习见 [Foundation 正文与练习映射](./exercise_map.md)，完成 `FQ05`。验证重点是输入检查、失败路径和最小回归用例。
