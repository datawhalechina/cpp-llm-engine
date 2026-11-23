# F12 CUDA 资源与异步执行

CUDA Kernel 只负责设备上的计算循环。一次完整执行还需要设备内存、主机到设备复制、Stream 排序、Event 观测和设备到主机复制。理解这些对象的所有权和完成时刻，比记住单个 API 更重要。

## 一条最小设备流水线

```text
Host input -> H2D -> Kernel -> D2H -> Host output
                 同一 Stream 保证提交顺序
                 Event 标记完成边界
```

| 对象 | 负责什么 | 生命周期要求 |
| --- | --- | --- |
| DeviceBuffer | 持有设备内存 | 最后一个使用它的设备任务完成后再释放 |
| PinnedHostBuffer | 提供可用于异步复制的主机内存 | 异步复制完成前保持地址和内容有效 |
| Stream | 排列同一执行队列中的任务 | 销毁前处理未完成工作或明确同步 |
| Event | 记录完成点和设备耗时 | 在目标 Stream 中记录，在读取前等待完成 |
| Workspace | 复用中间缓冲区 | 容量不足时增长，后续较小任务复用 |

异步 API 返回只表示任务已经提交，不表示输出已经能由 CPU 读取。调用者必须通过 Stream 同步、Event 或更高层完成通知建立可见性边界。

## 错误和资源边界

在任务入队前检查空指针、offset、容量和设备身份；Kernel launch 后还要检查 launch 错误。错误路径与正常路径使用同一 RAII 对象释放资源，避免在每个返回分支手写清理。

持久化 workspace 把“每次执行都分配”改为“对象创建或容量增长时分配”。它减少分配开销，但也要求对象明确服务哪个设备、Stream 或并发请求，避免多个未完成任务覆盖同一缓冲区。

## 验证与练习

先完成 [`FQ20 Device Pipeline`](https://github.com/datawhalechina/cpp-llm-engine/blob/main/exercises/foundation/problems/fq20_device_pipeline/README.md)，用 CPU 状态机固定 H2D、Kernel、D2H 和 workspace 复用语义。真实 CUDA 验证运行：

```bash
CUDA_HOME=/usr/local/cuda-12.8 bash code/cuda/scripts/run.sh
```

CPU 模拟用于理解顺序与所有权，不代替真实设备的 API、并发和性能测试。
