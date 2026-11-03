# 回归测试

回归测试保存已经通过的输入、输出和性能条件，用后续重复检查阻止旧问题再次出现。

## 1. 回归测试解决的问题

单次测试证明当前实现通过，回归测试检查后续改动是否保持相同结果。

回归样例重点覆盖以下长期风险：

- 旧 bug 被新改动重新引入
- 边界输入只在某次提交后失效
- 正常样例通过，但异常样例开始退化
- 性能看起来没坏，但长上下文或大 batch 开始不稳定

## 2. 分层测试体系

测试按发现问题的范围逐层扩大。下层没有通过时，不继续用端到端结果或性能数据解释问题。

| 层级 | 检查对象 | 典型证据 |
| --- | --- | --- |
| L0 对象契约 | 单个类型、函数和错误路径 | 命名单元测试、表驱动状态转换 |
| L1 不变量 | 一组操作前后必须保持的条件 | 进度单调、容量守恒、资源独占 |
| L2 组件协作 | Scheduler、Cache、Runtime 等组件事务 | 预留、提交、回滚和释放测试 |
| L3 跨章主线 | Loader 到 ServiceCore 的接口连接 | 集成测试和默认 CPU 回归 |
| L4 真实模型 | GGUF、token、logits、HTTP/SSE | TinyLlama golden 和真实服务测试 |
| L5 性能实验 | 固定软硬件条件下的执行表现 | benchmark、profiling 和历史结果 |

## 3. 回归测试和 e2e / benchmark 的分工

| 问题 | e2e correctness | regression test | benchmark |
| --- | --- | --- | --- |
| 目标 | 当前链路是否正确 | 旧问题是否被重新带回 | 固定条件下快多少 |
| 输入 | 典型样例 | 版本化样例集合 | 固定 prompt / batch / 硬件 |
| 输出 | 正确 / 不正确 | 是否偏离历史基线 | latency / throughput |
| 作用 | 证明这次实现没坏 | 证明历史问题没回归 | 证明性能变化是否值得 |

这三者不能互相替代。  
e2e 证明“这次能跑对”，benchmark 证明“这次快多少”，回归测试证明“这次没有把旧问题带回来”。

## 4. 回归测试应该覆盖什么

### golden corpus

golden corpus 是一组固定、版本化的测试样例，稳定覆盖最容易回退的关键路径：

- 典型 prompt
- 特殊 token 组合
- 固定长度输入
- 典型 batch 组合

这些样例一旦确定，就不应该随便改；如果模型版本、预处理逻辑或目标阈值变了，再显式更新。

### boundary shape

边界 shape 用来抓最容易被忽略的尺寸问题：

- 最小 batch
- 最大 batch
- 单 token 输入
- 长序列输入
- 非常规 shape 组合

这类样例经常能暴露 view、reshape、cache 复用和对齐假设上的问题。

### abnormal input

异常输入用于确认系统能稳定拒绝或降级：

- 空输入
- 非法 token
- 超长输入
- 截断后不完整的输入
- 不符合预期的 pad / truncation 组合

如果异常输入没有被单独检查，系统后面很容易在边界上产生静默错误。

### long-context degradation

长上下文退化测试关注的是：  
输入变长以后，系统是不是还保持合理行为。

常见关注点包括：

- KV Cache 是否稳定复用
- decode 阶段是否明显退化
- 内存占用是否异常增长
- 输出是否出现明显抖动

这类测试不一定看绝对性能，更看趋势是否符合预期。

## 5. 一个可维护的回归基线长什么样

可维护的回归基线通常有几个特征：

- 样例固定
- 版本可追踪
- 阈值明确
- 失败能定位到具体输入
- 结果能和历史提交对照

可维护的结果需要同时指出失败样例和失败阶段，而不只是“通过”或“失败”。

## 6. 最小基线矩阵

第一版不追求覆盖所有模型，先固定一组能在 CPU 上重复执行的样例：

| 类别 | 固定条件 | 主要断言 | 失败定位 |
| --- | --- | --- | --- |
| golden prompt | prompt、生成步数、采样策略 | token 序列或关键输出一致 | 输入 ID / 生成 step |
| boundary shape | batch=1、单 token、短序列、最大支持长度 | shape、cache length、退出码正确 | batch / sequence length |
| abnormal input | 空输入、非法 token、超长输入 | 明确拒绝或截断，不静默继续 | 输入校验阶段 |
| long context | 递增 prompt 长度 | cache、内存和输出趋势稳定 | context length / phase |
| benchmark baseline | 固定模型、硬件、batch 和 commit | latency / throughput 在阈值内 | phase / trace event |

验收顺序固定为：先跑局部测试，再跑端到端正确性，再跑固定 benchmark，最后把结果与历史基线比较。任何一层失败，都不应只用下一层的“通过”覆盖。

当前仓库的最小执行入口是 `npm run test:regression`。它串联 Foundation、Core、Engine、Chapter01-09 和多模态预处理，以命令退出码判断构建与 CTest 是否成功，并只对示例保留稳定的行为标记。`npm run test:release-contracts` 另用 Release 构建覆盖 Core、Chapter01-06、Chapter09、Engine 和多模态输入链的关键契约，防止测试依赖 Debug `assert`。测试数量会随能力增长，不作为成功条件。

`npm run test:check-assets` 在构建前检查测试资产：`tests/` 下的 C++/CUDA 测试源码必须被 CMake 引用；测试可执行目标必须调用 `cpp_llm_prepare_test`、注册到 CTest 并设置标签。该检查只验证测试是否接入体系，不替代测试执行和结果断言。

故障测试使用固定输入、手动时钟或显式失败执行器，不依赖随机调度。定位状态泄漏时，先运行下表中的最小测试，再运行所属章节回归。

| 对象 | 故障注入 | 必须保持的状态 | 最小测试 |
| --- | --- | --- | --- |
| Loader | 非法 magic、版本、shape、截断 payload | 加载失败后不暴露部分 tensor | `chapter01_tensor_loader_test` |
| BlockPool | 容量不足、越界提交、回退提交 | 容量守恒、块独占、原请求不变 | `chapter06_kv_block_pool_contract_test` |
| Scheduler | prefill 失败、缺少 decode 写槽 | 请求进入失败终态，临时预留回滚 | `chapter06_execution_transaction_test` |
| ServiceCore | 成功、模型失败、取消、超时交错 | inflight 归零、块全部归还、指标只计一次 | `chapter09_fault_sequence_test` |

## 7. 测试基础设施

`code/testing` 提供不受 `NDEBUG` 影响的命名检查。`CHECK` 记录失败并继续当前用例，`REQUIRE` 在前置条件失败后停止当前用例，避免后续索引或解引用掩盖最初错误；`CHECK_EQ` 和异常检查补充左右值及预期异常信息。公共 CMake 配置保证尚未迁移的 `assert` 在 Release 构建中继续执行。CTest 负责发现、筛选和运行测试；`unit`、`contract`、`component`、`integration` 等标签表达验证层级，模块名标签用于缩小失败范围。

默认回归检查教学主线，Release 契约回归专门检查编译优化不能关闭断言与契约测试：

```bash
npm run test:regression
npm run test:release-contracts
```

该入口使用 Release 配置验证测试框架、Core、Chapter 01 Loader、Tensor、MatMul、Attention/Decoder、Chapter 06 Runtime、
Chapter 09 ServiceCore 与 Engine 组合链。它不下载模型，也不运行 GPU 和性能实验。

新增能力时按以下顺序补证据：

1. 先写对象契约和错误路径。
2. 再写跨操作不变量，包括失败后的原子性。
3. 接入组件测试和跨章回归。
4. 正确性稳定后再登记真实模型与性能结果。

## 8. 和主线章节的关系

| 章节 | 已经做了什么 | 回归测试继续补什么 |
| --- | --- | --- |
| `chapter07_benchmark` | 正确性、benchmark、结果记录 | 长期回归基线和版本化样例 |
| `chapter06_generation` | runtime、KV Cache、scheduler | 长上下文和请求生命周期的稳定性 |
| `chapter01` 到 `chapter05` | 局部模块正确性 | 关键路径是否在后续改动中退化 |

单测和 e2e 提供当前正确性证据，版本化样例将这些证据保存为长期回归基线。

## 9. 如何使用回归测试参考页

- 先用 `chapter07_benchmark` 确认当前测试矩阵已经固定
- 再把历史上容易回退的样例整理成固定基线
- 然后在新改动后重复跑这些样例，观察是否偏离历史结果

## 核心结论

E2E 证明当前链路正确，回归测试检查后续版本是否保持该结果。
