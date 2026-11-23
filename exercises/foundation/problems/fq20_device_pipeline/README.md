# FQ20 Device Pipeline

## 任务

用 CPU 对象模拟一条有序设备流水线：主机到设备复制、核函数执行、Event 记录与等待、设备到主机复制。维护执行阶段和持久化 workspace，拒绝任务在途时复用资源，并支持失败后保留 workspace 重试。

## 完成标准

- `upload`、`launch`、`record_event`、`wait_event`、`download` 只能按顺序调用。
- 记录 `H2D -> kernel -> event -> wait -> D2H` 操作序列。
- Event 完成前拒绝下载和下一轮上传。
- workspace 只在容量不足时增长。
- 失败重置和第二轮较小输入复用第一次分配的容量。
- 非法执行顺序返回明确异常。

## 验证

复制并完成 starter 后，先运行自己的答案：

```bash
cp -r exercises/foundation/problems/fq20_device_pipeline/starter /tmp/fq20-answer
bash scripts/check_foundation_answer.sh /tmp/fq20-answer/main.cpp
```

运行 `bash exercises/foundation/run_all.sh`，重点检查 `fq20_device_pipeline` 的执行顺序和 workspace 复用。该练习验证设备语义；真实 CUDA API 由 `code/cuda` 专项入口验证。

## 提示

- L0：先定义上传、Kernel 在途、Event 已记录、完成和下载后的阶段。
- L1：每个方法先检查当前阶段，再修改数据与阶段。
- L2：失败只清除在途数据和阶段；workspace 容量保留给重试使用。

## 主线映射

对应 [`Foundation CUDA 资源`](/foundation/12_cuda_resource_lifecycle)、[`CUDA 代码架构`](/code/cuda) 和 [`Chapter09 性能证据`](/chapter09_performance/next/intro)。
