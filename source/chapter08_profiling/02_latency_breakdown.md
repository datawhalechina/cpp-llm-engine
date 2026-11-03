# 08.2 拆解端到端延迟

端到端延迟是多个阶段共同作用的结果。先区分首词元、后续词元和输出阶段，再决定需要检查 prompt 计算、逐词元解码、调度还是 transport。

## 把请求拆成可计时阶段

阶段起止点必须固定，同一事件不能同时计入两个阶段。离线 benchmark 从 dispatch 开始，在线指标还可能包含 admission 和 queue wait，比较前要先确认口径。

| 阶段 | 起止范围 | 常见检查对象 |
| --- | --- | --- |
| input | 请求进入执行到模型输入准备完成 | token、position、shape 和拷贝 |
| prefill | prompt 进入模型到首个 logits | 大矩阵、Attention 和初始 KV |
| decode | 每个输入 token 到下一组 logits | KV 读取、量化 Linear 和同步 |
| sampling | logits 到下一个 token | top-k/top-p、随机数和拷贝 |
| output | token 可用到响应写出 | 回调、序列化、SSE 和网络 |

## 用三个问题缩小范围

首词元慢通常先看 queue wait、input 和 prefill；后续词元慢先看 decode、KV 和 sampling；并发后才变慢则继续检查调度、资源竞争和 batch 形状。长 prompt 与长输出应使用不同 workload，不能从一次短样例同时推断两者。

Chapter07 的教学结果只用于练习阶段拆分。真实服务要同时记录 TTFT、TPOT/E2E、prefill/decode compute 和 Runtime overhead，再沿最重阶段进入 [`08.8 真实 TinyLlama 服务`](./08_real_service_profile.md)。

| 现象 | 优先查看 | 暂时不能直接断言 |
| --- | --- | --- |
| TTFT 上升 | queue wait、input、prefill | Attention kernel 一定变慢 |
| TPOT 上升 | decode、KV、sampling | 输出长度是唯一原因 |
| 并发后吞吐下降 | 调度、线程、batch、内存带宽 | Tensor batch 一定有效 |
| Output 阶段上升 | writer、序列化、背压 | 模型计算发生回退 |

## 验证与衔接

运行 `bash code/chapter07_benchmark/scripts/run.sh` 取得教学阶段结果。读完后应能为每项指标指出起止事件，并解释为什么离线 TTFT 与在线 TTFT 不能直接相减；下一步在 [`08.3`](./03_memory_bandwidth.md) 判断最重阶段是否受数据移动和内存布局影响。
