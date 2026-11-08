# FQ06 CPU MatMul

## 任务

实现 row-major 的 CPU reference MatMul：`A[M,K] × B[K,N] = C[M,N]`。

## 完成标准

- 检查输入 buffer 长度与 shape 是否匹配。
- 使用 `O(MKN)` 三重循环计算结果。
- 覆盖 `1×1`、单行、单列和普通矩阵。
- 与固定参考答案对照，并说明布局约定。

## 验证

复制并完成 starter 后，先运行自己的答案：

```bash
cp -r exercises/foundation/problems/fq06_cpu_matmul/starter /tmp/fq06-answer
bash scripts/check_foundation_answer.sh /tmp/fq06-answer/main.cpp
```

运行 `bash exercises/foundation/run_all.sh`，重点测试 `fq06_cpu_matmul`。

## 提示

- L0：先写出 `A[row * K + inner]` 和 `B[inner * N + col]`。
- L1：先验证 shape，再分配 `M * N` 的零初始化输出。
- L2：增加不规则矩阵和非法 buffer 长度测试。

## 主线映射

对应 [`chapter03_matmul/03_cpu_reference`](/chapter03_matmul/03_cpu_reference)。
