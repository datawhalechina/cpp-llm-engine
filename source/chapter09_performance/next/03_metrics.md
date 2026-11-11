# 09.3 选择能回答问题的指标

不同指标回答不同问题。单请求延迟不能代替多请求吞吐，平均延迟也不能说明尾部请求是否被长期等待。

| 指标 | 计算范围 | 主要回答 |
| --- | --- | --- |
| 端到端墙钟时间（end-to-end wall-clock time） | Host 入口开始到结果可用的真实经过时间 | 学习者或调用方实际等待多久 |
| 总延迟 | 请求进入到结束 | 一个请求多久完成 |
| TTFT | 请求进入到首个输出词元 | 用户多久看到首次响应 |
| ITL | 相邻输出词元的时间间隔 | 流式输出是否平稳 |
| 吞吐 | 完成词元数 / 总时间 | 系统单位时间完成多少工作 |
| 完成时间分布 | 多请求各自结束时间 | 公平性和尾延迟是否恶化 |
| 峰值内存 | 实验期间最大占用 | 当前模型和并发是否能放入设备 |

指标名称必须带清楚的统计范围。例如 `decode_tokens_per_second` 与端到端 `tokens_per_second` 不能当成同一个数字比较。

墙钟时间也称经过时间（elapsed time）。它按真实时钟计算，期间的 CPU/GPU 计算、同步、数据搬运、调度和等待都会推进计时。CPU time 只统计进程实际占用 CPU 的时间，CUDA Event 只测同一 stream 上两个事件之间的设备工作；两者都不能单独替代端到端墙钟时间。

## 先区分 Prefill 和 Decode

Prefill 一次处理多个输入词元，常表现为较大的矩阵计算；Decode 每步产生一个新词元，更容易受权重读取、K/V 访问和调度开销影响。只记录总时间会掩盖两类路径的变化方向。

| 现象 | 优先核对 |
| --- | --- |
| TTFT 上升、ITL 稳定 | Prompt 长度、Prefill 算子与排队 |
| TTFT 稳定、ITL 上升 | Decode、K/V 访问与每步调度 |
| 吞吐上升、尾延迟恶化 | Batch 大小、公平性与等待时间 |
| 延迟不变、内存上升 | K/V 容量、临时 Tensor 与缓存释放 |

## 把指标放入同一样本

<a href="https://github.com/datawhalechina/cpp-llm-engine/blob/main/code/chapter09_performance/include/chapter09_performance/evidence_record.h"><code>MeasurementSample</code></a> 在一轮测量中同时保存总时间、Prefill、Decode、TTFT、ITL、吞吐和峰值内存。这使同一请求的阶段数据共享 workload 和环境身份，不需要在多个文件中猜测它们是否来自同一次运行。

运行 `bash code/chapter09_performance/scripts/run.sh` 后，`chapter09_evidence_contract_test` 应确认所有样本都是非负有限值，且总时间和吞吐为正。指标语义固定后，`09.4` 再对重复样本进行聚合。
