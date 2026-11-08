# chapter09_performance 代码导航

本章代码把 Benchmark 原始测量、统计聚合和优化决策分成三个可独立验证的对象。

## 阅读顺序

1. 先看 `evidence_record.h/.cpp`，确认一份测量需要哪些身份和正确性字段
2. 再看 `statistics.h/.cpp`，跟踪原始样本怎样生成聚合结果
3. 最后看 `comparison.h/.cpp`，观察错误、不可比和性能回退怎样分流

## 证据与统计

| 文件 | 作用 |
| --- | --- |
| <a href="/_code/chapter09_performance/include/chapter09_performance/evidence_record.h"><code>evidence_record.h</code></a> | 环境、工作负载、样本和证据记录接口 |
| <a href="/_code/chapter09_performance/src/evidence_record.cpp"><code>evidence_record.cpp</code></a> | 完整性校验、聚合和 JSON 输出 |
| <a href="/_code/chapter09_performance/include/chapter09_performance/statistics.h"><code>statistics.h</code></a> | 统计结果接口 |
| <a href="/_code/chapter09_performance/src/statistics.cpp"><code>statistics.cpp</code></a> | 样本范围、均值和中位数计算 |

## 对比与验证

| 文件 | 作用 |
| --- | --- |
| <a href="/_code/chapter09_performance/include/chapter09_performance/comparison.h"><code>comparison.h</code></a> | 阈值和四类决策接口 |
| <a href="/_code/chapter09_performance/src/comparison.cpp"><code>comparison.cpp</code></a> | 可比性检查与保护指标判断 |
| <a href="/_code/chapter09_performance/tests/evidence_contract_test.cpp"><code>evidence_contract_test.cpp</code></a> | 测量记录契约测试 |
| <a href="/_code/chapter09_performance/tests/comparison_test.cpp"><code>comparison_test.cpp</code></a> | KEEP、REJECT、NOT_COMPARABLE 和 INVALID 测试 |
| <a href="/_code/chapter09_performance/scripts/run.sh"><code>scripts/run.sh</code></a> | Chapter09 独立构建与 CTest 入口 |
| <a href="/_code/chapter09_performance/scripts/import_legacy_evidence.mjs"><code>import_legacy_evidence.mjs</code></a> | 将保留逐轮样本的历史结果转换为统一证据 |
| <a href="/_code/chapter09_performance/scripts/compare_evidence_pair.mjs"><code>compare_evidence_pair.mjs</code></a> | 将 JSON 规范化后交给 C++ 比较器 |
| <a href="/_code/chapter09_performance/scripts/run_real_evidence.sh"><code>run_real_evidence.sh</code></a> | 生成真实服务、Profiling 或 A/B 证据 |

当前真实 TinyLlama 原始样本保存在 <a href="/_code/chapter09_performance/results/raw/tinyllama-service-optimized-avx2-sequential.json"><code>results/raw/</code></a>，统一 evidence v2 保存在 <a href="/_code/chapter09_performance/results/evidence/tinyllama_service_core_cpu-optimized-avx2-sequential.json"><code>results/evidence/</code></a>；`legacy/` 只保留重构前资产供追溯。

当前阶段结果见 <a href="/_code/chapter09_performance/results/profile/tinyllama-decoder-scalar.txt"><code>tinyllama-decoder-scalar.txt</code></a>；AVX2 顺序/并行请求的归档决策见 <a href="/_code/chapter09_performance/results/evidence/tinyllama_service_core_cpu-avx2-dispatch-decision.json"><code>dispatch-decision.json</code></a>。
