# FQ08 Mini Tensor

## 任务

组合 ownership、slice 和索引访问，实现一个拥有共享 storage、可以创建 view 的最小 Tensor。

## 完成标准

- 根 Tensor 管理底层 storage。
- slice 只创建共享 storage 的 view，不复制数据。
- view 修改能反映到根 Tensor。
- 根 Tensor 销毁后，仍存活的 view 不访问悬空内存。
- 越界索引和非法 slice 有明确失败行为。

## 验证

复制并完成 starter 后，先运行自己的答案：

```bash
cp -r exercises/foundation/problems/fq08_mini_tensor/starter /tmp/fq08-answer
bash scripts/check_foundation_answer.sh /tmp/fq08-answer/main.cpp
```

运行 `bash exercises/foundation/run_all.sh`，重点测试 `fq08_mini_tensor`；再用 UBSan/ASan 检查生命周期。

## 提示

- L0：区分 storage owner、view offset 和 view length。
- L1：使用共享所有权保存数组生命周期。
- L2：增加“根对象先销毁、view 后使用”的测试。

## 主线映射

对应 [`foundation/09_minimal_cpp_cuda_project`](/foundation/09_minimal_cpp_cuda_project)、[`chapter01 Tensor 对象`](/chapter01_tensor_loader/01_tensor_object) 和 [`chapter02_tensor_ops/03_slice`](/chapter02_tensor_ops/03_slice)。
