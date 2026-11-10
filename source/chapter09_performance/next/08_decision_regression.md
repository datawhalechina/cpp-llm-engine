# 09.8 把实验结果变成保留或回退决策

性能决策由正确性、可比性和阈值共同决定。吞吐提升不自动覆盖延迟、内存或公平性回退；主指标和保护指标需要在实验前写清楚。

| 决策 | 条件 | 后续动作 |
| --- | --- | --- |
| `KEEP` | 正确、可比，主指标改善且保护指标未越界 | 保留修改并登记证据 |
| `REJECT` | 正确但性能越过回退阈值 | 回退或重新设计 |
| `NOT_COMPARABLE` | workload 或关键环境不一致 | 修正实验后重跑 |
| `INVALID` | 任一侧正确性或记录校验失败 | 先修复功能问题 |

## 性能回归检查

持续集成适合运行短小、稳定的功能与 schema 契约；真实性能容易受到共享机器影响，不应仅凭一次 CI 计时阻断提交。稳定环境可以使用宽阈值发现明显回退，发布结论仍需专用机器上的重复实验。

| 层级 | 自动化内容 | 结论强度 |
| --- | --- | --- |
| 每次提交 | 正确性、记录 schema、统计与比较器测试 | 数据链路没有破坏 |
| 定期运行 | 固定机器、固定 workload、多轮样本 | 发现趋势与明显回退 |
| 发布实验 | 环境快照、完整样本、阶段 Profiling、A/B 记录 | 支持优化保留决策 |

实验资产登记表保存结论和结果路径，正文解释方法。当前 AVX2 顺序/并行请求决策见 <a href="../../_code/chapter09_performance/results/evidence/tinyllama_service_core_cpu-avx2-dispatch-decision.json"><code>dispatch-decision.json</code></a>；文件引用两份 evidence，不复制原始样本。正文中的示例数字不能替代可下载、可复查的原始结果。

局部 Kernel 或专家子层通过门禁后，还要沿真实 Decoder、Generation Runtime、Scheduler 和 Service 逐层复核。每一层都重新固定计时范围和保护指标；局部加速可以保留为候选，但只有端到端证据通过时才能改变默认执行路径。适用范围矩阵再说明该决定覆盖哪些 batch、上下文和 Decode 步数，不能把一个单点结论推广到全部请求。

## 验证决策回归

运行 `bash code/chapter09_performance/scripts/run.sh`。`chapter09_comparison_test` 应锁定四类决策，`chapter09_versioned_evidence_test` 应确认已归档结果仍符合当前 schema，`chapter09_real_evidence_bridge_test` 应确认真实服务样本可以进入同一比较器。

决策文件只回答指定 workload 下是否保留修改。服务延迟的起点、排队与网络范围由 Chapter10 固定；下一篇先将本章证据对象放回成熟框架的工具链。
