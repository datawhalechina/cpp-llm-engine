# 07.6 从请求独占缓存走向生产扩展

当前实现为每个请求分配独占物理块，适合验证分页地址、事务和真实模型结果。成熟系统还要处理跨请求前缀复用、设备内存管理、批量元数据更新和融合 Attention kernel。本节只建立这些扩展与当前对象的对应关系，不把未实现能力写成可运行结果。

## 前缀复用增加什么

两个请求拥有相同 prompt 时，独占缓存会重复计算并保存 K/V。前缀缓存为已计算的词元序列建立索引，让新请求引用只读前缀块，只计算未命中的后缀。

支持共享后，系统需要前缀索引、共享块引用计数、执行期间的锁定状态、命中长度和淘汰策略。多个页表可以引用同一只读前缀块，但请求写入私有后缀前不能修改共享块。Radix 树负责最长前缀匹配，Chapter08 的调度器决定何时执行命中的请求；缓存索引与调度策略仍是两个对象。

当前 `KvBlockPool` 没有跨请求前缀索引、共享引用和淘汰策略。若后续实现，测试必须证明共享前缀只计算一次、共享块不会被覆盖、单个请求结束不会提前释放块，以及容量压力下只淘汰未锁定前缀。

| 新对象 | 回答的问题 |
| --- | --- |
| 前缀索引 | 新请求与哪些已缓存词元匹配 |
| 引用计数 | 物理块何时不再被任何请求使用 |
| 锁定状态 | 调度和执行期间哪些块不能淘汰 |
| 淘汰策略 | 容量不足时先回收哪些未使用前缀 |
| 命中长度 | Prefill 从哪个位置继续计算 |

## 教学对象怎样对应成熟系统

阅读框架源码时继续追踪三个问题：谁分配物理块，谁保存真实 K/V，Attention 怎样获得地址信息。CPU 与 GPU 可以使用不同数据布局和执行内核，但都要维护同一组请求所有权和提交语义。

| 教程对象 | 成熟系统中的方向 | 生产环境继续处理 |
| --- | --- | --- |
| `BlockTable` | 请求到物理 K/V 页的映射 | 设备元数据、批量更新、跨步持久化 |
| `KvBlockPool` | Cache manager / block allocator | 显存容量、抢占、交换、并发安全 |
| `CpuPagedKvCache` | Cache engine / Paged Attention 数据面 | GPU kernel、布局优化、多数据类型 |
| 连续与分页对照 | Reference / optimized backend | 数值误差、吞吐、回退策略 |
| 请求独占块 | Prefix / Radix Cache | 共享引用、锁定、命中、淘汰 |

## 继续阅读框架源码

llama.cpp 适合观察单机 Context 与真实模型 K/V；vLLM 适合追踪缓存管理、块表和 Attention metadata；Mini-SGLang 与 SGLang 展示 Radix 前缀复用和调度协作；TensorRT-LLM 展示 GPU K/V 与高性能 Attention kernel。Chapter08 接下来只通过 `KvCacheTransaction` 预留和回收资源，不直接读取 K/V 数值。

| 项目卡 | 重点追踪 |
| --- | --- |
| [llama.cpp](/reference/ecosystem/llama_cpp) | Context 怎样拥有并增长真实模型 K/V |
| [vLLM](/reference/ecosystem/vllm) | Cache manager、块表与 Attention metadata |
| [nano-vLLM](/reference/ecosystem/nano_vllm) | 紧凑代码中的页表与缓存分配骨架 |
| [Mini-SGLang](/reference/ecosystem/mini_sglang) | Radix Cache 的匹配、引用和释放 |
| [SGLang](/reference/ecosystem/sglang) | 前缀缓存、多级缓存与调度扩展 |
| [TensorRT-LLM](/reference/ecosystem/tensorrt_llm) | GPU K/V 与 Attention kernel 的工程入口 |
