# FQ02 Tensor Shape

## 任务

实现 `numel(shape)`，计算 Tensor 的元素数量，并拒绝空 shape 或包含零维度的输入。

## 完成标准

- `{2, 3, 4}` 返回 `24`。
- `{1}` 返回 `1`。
- 空 shape 和零维度抛出 `std::invalid_argument`。
- 乘积溢出策略至少在文档中说明。

## 验证

复制并完成 starter 后，先运行自己的答案：

```bash
cp -r exercises/foundation/problems/fq02_tensor_shape/starter /tmp/fq02-answer
bash scripts/check_foundation_answer.sh /tmp/fq02-answer/main.cpp
```

运行 `bash exercises/foundation/run_all.sh`，重点测试 `fq02_tensor_shape`。

## 提示

- L0：先区分 rank、shape 和 numel。
- L1：使用累乘，不要把维度数量误当作元素数量。
- L2：补充空输入、零维度和大维度测试。

## 主线映射

对应 [`chapter01 Tensor 对象`](/chapter01_tensor_loader/01_tensor_object)。
