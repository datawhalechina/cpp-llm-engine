# FQ18 Numerical Evidence

## 任务

实现最大绝对误差、均方根误差、余弦相似度和数值稳定的 Softmax。使用固定输入比较参考结果与候选结果，并明确拒绝 shape 不一致或零向量等无效输入。

## 完成标准

- 三种误差指标都检查输入长度和空输入。
- 稳定 Softmax 先减去最大值，再计算指数与归一化。
- 概率和在浮点容差内等于 1，较大的 logit 得到较大的概率。
- 测试不使用浮点逐位相等代替容差判断。
- 非法输入返回明确异常。

## 验证

复制并完成 starter 后，先运行自己的答案：

```bash
cp -r exercises/foundation/problems/fq18_numerical_evidence/starter /tmp/fq18-answer
bash scripts/check_foundation_answer.sh /tmp/fq18-answer/main.cpp
```

运行 `bash exercises/foundation/run_all.sh`，重点检查 `fq18_numerical_evidence` 的误差指标、稳定 Softmax 和失败输入。

## 提示

- L0：最大绝对误差关注最坏元素，均方根误差关注整体偏差。
- L1：余弦相似度比较方向，需要同时计算点积和两个向量的二范数。
- L2：Softmax 的指数对平移不敏感，减去最大值可以避免大 logit 溢出。

## 主线映射

对应 [`Foundation 数值验证`](/foundation/11_numerical_validation)、[`Chapter03 PyTorch 对照`](/chapter03_matmul/05_test_with_pytorch) 和 [`Chapter09 性能证据`](/chapter09_performance/next/intro)。
