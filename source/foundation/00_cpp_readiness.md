# F00 C++ 就绪自测

Foundation 讲解推理引擎需要的 C++ 工程能力，不替代完整的 C++ 入门课程。开始题库前，先用下面的清单判断是否需要补齐语法基础。

## 自测范围

阅读下面的程序，尝试在不运行的情况下说出输出，并解释 `const&` 为什么没有复制 `values`：

```cpp
#include <iostream>
#include <string>
#include <vector>

struct Request {
  int id = 0;
  std::string prompt;
};

int sum(const std::vector<int>& values) {
  int result = 0;
  for (const int value : values) {
    result += value;
  }
  return result;
}

int main() {
  const Request request{7, "hello"};
  const std::vector<int> tokens{1, 2, 3};
  std::cout << request.id << ' ' << sum(tokens) << '\n';
}
```

用 C++17 编译并运行：

```bash
c++ -std=c++17 -Wall -Wextra -Wpedantic main.cpp -o readiness
./readiness
```

预期输出是 `7 6`。继续学习前，应能完成这些操作：

- 编写函数、循环、条件分支和简单的 `struct` 或 `class`。
- 使用 `std::string`、`std::vector` 和 `std::deque` 保存数据。
- 区分值、引用、指针和 `const` 引用，知道对象存在有效生命周期。
- 看懂 `enum class`、构造函数和异常抛出。
- 用一条编译命令生成并运行单文件程序。

## 如何选择入口

| 自测结果 | 建议入口 | 原因 |
| --- | --- | --- |
| 能解释并改写示例 | 直接阅读 [F01 C++ 基础](./01_cxx_basics.md) | 可以集中学习所有权、RAII 和工程约束 |
| 能运行示例，但引用、类或容器不熟 | 先按弱基础路线完成 `FQ02 -> FQ05 -> FQ07` | 从数据、边界和 FIFO 行为进入，避免一开始同时处理资源所有权 |
| 无法独立编译示例 | 先补函数、类、标准容器和编译命令 | 后续题目默认具备这些语法能力 |

自测页不设题号，也不计入 Foundation 的 22 道练习。它只负责选择入口；完成所需基础后，仍使用题目 README 中的命令验收代码。
