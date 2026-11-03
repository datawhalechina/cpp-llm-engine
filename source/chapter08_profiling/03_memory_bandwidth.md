# 08.3 定位内存与带宽瓶颈

推理性能不仅取决于浮点运算量，也取决于权重、激活和 KV 在内存中的布局与移动方式。量化降低存储不等于端到端一定加速，分页管理 block ID 也不等于 Paged KV 数据路径已经完成。

## 拆分四类内存成本

分析时先记录对象大小、生命周期、访问顺序和所在设备，再判断应该减少容量、复制、分配还是等待。

| 成本 | 观察对象 | 当前教程入口 |
| --- | --- | --- |
| Weight | dtype、packed block、是否重复持有 | Chapter05 量化 Linear |
| Activation | 中间 Tensor、连续化和 workspace | Chapter02 View、Chapter03–05 backend |
| KV Cache | 每层 K/V、增长、定位和释放 | Chapter06 连续 KV 与 BlockTable |
| Transfer | Host/Device copy、stream 和同步 | `code/cuda` memory/stream 测试 |

## 从信号进入证据

“batch 变大但吞吐不升”“量化后体积下降但延迟不降”“decode 每步重复分配”都只是调查信号。需要继续用分配次数、复制字节、带宽计数器或 A/B 证明原因。

当前 scalar batch 的 `0.8541x` 结果说明复用权重遍历没有自动转化为收益。分段计时显示输入整理与输出拆分合计仅占 `0.0184%`，主要时间仍在 scalar batch kernel；BlockTable 目前只管理物理 ID，真实模型仍使用连续 K/V。两者都不能用概念上的带宽收益代替实测。

| 当前对象 | 已有证据 | 继续补什么 |
| --- | --- | --- |
| Q4_K/Q6_K AVX2 | 单变量 Decoder/ServiceCore A/B | 其他 CPU、shape 和 batch |
| Scalar Linear batch | 正确性通过但性能回退 | 专用 SIMD batch 与热点证据 |
| 连续模型 K/V | 多请求隔离和固定 logits | 长上下文容量与带宽 |
| BlockTable | reserve/commit/rollback/release | 物理 KV payload 与 Paged Attention |

## 区分载荷、已提交缓存和进程峰值

`EXP-CPU-02` 在双请求结束、资源释放之前读取 Model 与两个 Context，并在进程退出前读取峰值常驻集。三个数字回答的问题不同：权重载荷是已加载 GGUF Tensor 的 payload 之和；KV 是当前请求按层实际提交的 K/V 向量容量；峰值 RSS 是操作系统观察到的整个进程高水位，还包含 tokenizer、对象元数据、临时工作区、分配器和运行库。

这三个值不能直接相加或互相相减。当前短请求只提交 13 个 token，因此 KV 远小于权重；更长上下文需要按相同公式重新测量，物理分页 KV 接入后也要替换连续向量的统计入口。

| 口径 | `EXP-CPU-02` 结果 | 能支持的判断 |
| --- | ---: | --- |
| 模型权重载荷 | `667078656 B` | 当前 Model 实际持有的 Tensor payload 规模 |
| 双请求已提交 KV | `585728 B` | 9 token 与 4 token 在 22 层连续 K/V 中的实际字节数 |
| 进程峰值 RSS | `652.82 MiB` | 本次进程生命周期的内存高水位，不是模型独占内存 |

## 验证与衔接

读完后应能区分“占用多少内存”“搬运多少字节”和“等待多长时间”。具体结构见 [`06.12 Block KV Cache`](../chapter06_generation/12_block_kv_cache.md)，硬件级带宽结论需要 `perf`、Nsight Compute 或等价计数器支持。
