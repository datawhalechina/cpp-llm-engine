# FQ03 Tensor View

## 任务

实现一个不拥有底层数据的连续 view。view 只保存指针、长度和 offset，修改 view 必须反映到原始 storage。

## 完成标准

- view 不复制底层数组。
- `offset + index` 的寻址正确。
- 越界访问被拒绝。
- 原始 storage 和 view 的修改互相可见。

## 验证

复制并完成 starter 后，先运行 `bash scripts/check_foundation_answer.sh /tmp/fq03-answer/main.cpp`。

运行 `bash exercises/foundation/run_all.sh`，重点测试 `fq03_tensor_view`。

## 提示

- L0：先画出 storage、offset 和 view length 的关系。
- L1：明确 view 不负责释放 storage。
- L2：测试首元素、末元素、空窗口和越界访问。

## 主线映射

对应 [`chapter02_tensor_ops/01_shared_ptr_memory`](/chapter02_tensor_ops/01_shared_ptr_memory) 和 [`chapter02_tensor_ops/03_slice`](/chapter02_tensor_ops/03_slice)。
