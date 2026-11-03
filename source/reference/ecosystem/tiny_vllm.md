# Tiny-VLLM C++/CUDA 纵向实现

## 为什么读

[Tiny-VLLM](https://github.com/jmaczan/tiny-vllm) 用 C++ 和 CUDA 串起权重读取、RMSNorm、RoPE、GEMM、Attention、KV 和分页管理，适合观察一条执行路径如何从基础 kernel 逐步增长。

## 重点看什么

- 一个模型结构怎样直接映射到 C++ 数据和 CUDA 调用
- kernel 启动、设备内存与 cuBLAS 如何进入同一条执行路径
- Paged KV 的地址映射怎样影响 Attention 输入

Tiny-VLLM 采用线性模块目录；本教程额外拆出 Loader、Tensor/View、验证、Profiling 和 Serving，用于连接算法实现与完整服务链路。

## 对应教程章节

- Foundation：CUDA thread/block、内存和最小 kernel
- Chapter03-04：GEMM、RoPE、Attention 与 backend
- Chapter05-06：量化执行、KV 和分页管理
- `code/cuda`：单卡 CUDA 纵向切片
