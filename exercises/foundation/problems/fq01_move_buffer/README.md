# FQ01 Move-only Buffer

## 任务

实现一个只允许移动、不允许复制的连续 `float` 缓冲区。对象必须独占资源，并在移动后将源对象置为空状态。

## 完成标准

- 拷贝构造和拷贝赋值被禁用。
- 移动构造和移动赋值转移指针与长度。
- 移动后的源对象长度为 `0`。
- 资源释放只发生一次。

## 验证

复制并完成 starter 后，先运行自己的答案：

```bash
bash scripts/check_foundation_answer.sh /tmp/fq01-answer/main.cpp
```

参考实现：`../../fq01_move_buffer.cpp`。完成后运行：

```bash
bash exercises/foundation/run_all.sh
```

重点测试 `fq01_move_buffer`，再使用 `-DENABLE_ASAN=ON` 检查资源释放。

## 提示

- L0：先写出资源所有权和空状态的类不变量。
- L1：使用 `std::unique_ptr<float[]>` 管理数组。
- L2：分别测试移动构造、移动赋值和自赋值。

## 主线映射

对应 [`foundation/01_cxx_basics`](/foundation/01_cxx_basics) 和 [`chapter02_tensor_ops/01_shared_ptr_memory`](/chapter02_tensor_ops/01_shared_ptr_memory)。
