# FQ10 Vector Add / CUDA Mapping

## 任务

先实现 CPU 版 `vector_add` 作为正确性参考，再把每个输出元素映射到一个 CUDA thread。默认 CTest 只验证 CPU 参考实现，不要求本机安装 CUDA。

## 完成标准

- 相同长度的输入逐元素相加。
- 长度不一致时明确失败。
- CPU 参考结果可作为 CUDA kernel 的对照。
- 能说明 `i < n` 边界检查、grid-stride loop 和 host/device 数据搬运的位置。

## 验证

复制并完成 CPU starter 后，先运行自己的答案：

```bash
cp -r exercises/foundation/problems/fq10_vector_add/starter /tmp/fq10-answer
bash scripts/check_foundation_answer.sh /tmp/fq10-answer/main.cpp
```

运行 `bash exercises/foundation/run_all.sh`，重点测试 `fq10_vector_add`。有 CUDA 环境时，再将 `starter/main.cu` 接入独立 CUDA 工程，对照 CPU 结果验证。

## 提示

- L0：先写 CPU 循环和输入长度检查。
- L1：CUDA kernel 中使用 `blockIdx.x * blockDim.x + threadIdx.x` 计算索引。
- L2：增加非整除 block size 和空向量测试，再比较 CPU/CUDA 结果。

## 主线映射

对应 [`chapter03 MatMul 与计算后端`](/chapter03_matmul/intro) 和 [`foundation CUDA 环境基础`](/foundation/03_cuda_environment_basics)。
