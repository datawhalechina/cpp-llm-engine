# 07.10 对照成熟框架的缓存引擎

本章用请求页表、物理块池和 CPU 分页数据面拆开缓存管理。阅读成熟框架时继续追踪这三个问题：谁分配块，谁保存真实 K/V，Attention 怎样得到地址信息。

## 对象映射

| 教程对象 | 成熟系统中的设计方向 | 继续增加的工程问题 |
| --- | --- | --- |
| `BlockTable` | 请求到物理 KV 页的映射 | 设备元数据、批量更新、跨执行步持久化 |
| `KvBlockPool` | KV cache manager 或 block allocator | 设备容量、抢占、交换与并发安全 |
| `CpuPagedKvCache` | cache engine 和 Paged Attention 数据面 | GPU kernel、布局优化和多数据类型 |
| 连续/分页对照 | reference backend 与优化 backend | 数值误差、吞吐和回退策略 |
| 请求独占块 | prefix cache 或 Radix Cache | 共享引用、锁定、命中和淘汰 |

llama.cpp 适合观察单机 Context 和连续缓存怎样服务真实模型；vLLM 适合追踪分页 KV、缓存管理和 Attention metadata；Mini-SGLang 与 SGLang 适合观察 Radix 前缀复用怎样与调度协作；TensorRT-LLM 适合观察高性能 GPU 执行如何组织 KV Cache 和 Attention kernel。

## 教学实现与生产实现

| 本章已有证据 | 生产系统继续处理 |
| --- | --- |
| CPU 固定容量物理块池 | GPU 显存容量、异步分配和跨设备管理 |
| 请求独占 BlockTable | 前缀共享、引用计数、抢占和交换 |
| 单 query 的分页 GQA 参考 | 批量 query、融合 kernel 和多精度布局 |
| 同步 reserve/write/commit | 并发执行、事件同步和失败恢复 |
| 释放后清零并复用 | 高吞吐回收、延迟清理和安全隔离 |

## 继续阅读源码

| 项目卡 | 在本章中重点追踪 |
| --- | --- |
| [llama.cpp](/reference/ecosystem/llama_cpp) | Context 怎样拥有并增长真实模型 K/V |
| [vLLM](/reference/ecosystem/vllm) | Cache manager、block table 与 Attention metadata 怎样连接 |
| [nano-vLLM](/reference/ecosystem/nano_vllm) | 紧凑代码怎样表达页表和缓存分配骨架 |
| [Mini-SGLang](/reference/ecosystem/mini_sglang) | Radix Cache 的匹配、引用和释放 |
| [SGLang](/reference/ecosystem/sglang) | 前缀缓存、多级缓存与调度的生产扩展 |
| [TensorRT-LLM](/reference/ecosystem/tensorrt_llm) | GPU KV Cache 与 Attention kernel 的工程入口 |
