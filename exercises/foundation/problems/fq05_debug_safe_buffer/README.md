# FQ05 Safe Buffer Access

## 任务

为连续缓冲区增加边界检查，并用一个最小测试验证非法访问会以明确的异常失败，而不是产生未定义行为。

## 完成标准

- 合法下标返回正确值。
- 越界下标抛出 `std::out_of_range`。
- 测试覆盖最后一个合法元素和第一个非法元素。
- UBSan/ASan 构建不报告问题。

## 验证

复制并完成 starter 后，先运行 `bash scripts/check_foundation_answer.sh /tmp/fq05-answer/main.cpp`。

运行普通题库和 UBSan 题库：

```bash
bash exercises/foundation/run_all.sh
cmake -S exercises/foundation -B exercises/foundation/build-ubsan -DCMAKE_CXX_FLAGS=-fsanitize=undefined
cmake --build exercises/foundation/build-ubsan
ctest --test-dir exercises/foundation/build-ubsan --output-on-failure
```

## 提示

- L0：先定义合法下标范围。
- L1：异常路径也要有测试。
- L2：对比 `operator[]`、`at()` 和自定义检查的行为边界。

## 主线映射

对应 [`foundation/04_debugging_testing_basics`](/foundation/04_debugging_testing_basics)。
