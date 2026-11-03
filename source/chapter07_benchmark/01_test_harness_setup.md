# 07.1 固定测试入口与失败信息

性能数据必须建立在可重复的正确性检查上。本节先查看 Chapter07 怎样把小型计算、结果渲染和多请求行为接入 CTest，并区分这些测试能够证明的范围。

## 测试入口与用例组织

`chapter07_benchmark_test` 使用轻量测试框架逐个执行具名函数，失败时报告用例名称。它验证速率计算、阶段占比、摘要文本、事件轨迹和 JSON 字段，不依赖外部模型文件。

| 测试对象 | 入口 | 可观察证据 |
| --- | --- | --- |
| 指标计算 | `make_result`、`tokens_per_second`、`phase_share` | 固定输入得到固定数值 |
| 结果格式 | `render_summary`、`render_trace_json` | 必需字段和事件名称存在 |
| 教学多请求 | `chapter07_multi_request_benchmark` | 请求完成、生成数量和缓存隔离成立 |
| Chapter06 行为 | Chapter06 的 CTest | prefill/decode、终态和调度失败路径成立 |

测试文件与被测对象保持直接对应，避免用一个“大而全”的端到端用例掩盖具体失败位置。小型测试使用代码内构造的数据；真实 GGUF 模型检查通过独立脚本执行，不进入默认 CTest。

## 从失败名称定位对象

先看 <a href="../_code/chapter07_benchmark/tests/chapter07_benchmark_test.cpp"><code>chapter07_benchmark_test.cpp</code></a> 中注册的用例，再看 <a href="../_code/chapter07_benchmark/src/test_helpers.cpp"><code>test_helpers.cpp</code></a> 中的固定阶段数据。两者分别回答“检查什么”和“用什么输入检查”。

例如 `tokens_per_second` 失败时，应先检查单位换算；`multi_request_benchmark` 失败时，应检查完成请求数、生成词元数和每个请求的缓存长度，而不是先分析机器快慢。

默认测试没有验证真实模型 logits，也没有衡量机器性能。固定格式样例用于检查报告结构；实际耗时由 `runtime_benchmark` 和 `multi_request_benchmark` 产生。

## 验证与衔接

```bash
bash code/chapter07_benchmark/scripts/run.sh
```

通过条件是 CTest 全部成功，且终端能够用具体测试名称定位失败。测试入口稳定后，继续阅读 [`07.2 端到端正确性`](./02_e2e_correctness.md)，区分教学 Runtime、跨章真实模型和性能测量的证据。
