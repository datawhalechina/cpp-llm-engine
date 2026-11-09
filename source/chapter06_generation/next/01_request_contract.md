# 06.1 建立请求与模型执行契约

模型已经能接收词元并产生 logits，但生成过程会跨越多次模型调用。代码需要保存输入、已生成词元、当前位置和结束状态，同时把模型计算与生成控制流分开。Chapter06 用请求（`Request`）、生成配置（`GenerationConfig`）和模型执行器（`RequestModelExecutor`）固定这组交接数据。

## 请求保存哪些状态

请求创建时接收 prompt，随后只通过明确的状态转换提交进度。<a href="/_code/chapter06_generation/include/chapter06_generation/request.h"><code>request.h</code></a> 暴露读取接口和状态转换，<a href="/_code/chapter06_generation/src/request.cpp"><code>request.cpp</code></a> 保证终态不会再次恢复为运行态。

| 数据 | 代码表示 | 变化时机 |
| --- | --- | --- |
| 输入词元 | `prompt()` | 构造请求时固定 |
| Prefill 进度 | `prefill_offset()` | 一段 prompt 成功提交，或采用已验证的缓存前缀后增加 |
| 已生成词元 | `generated()` | 采样结果成功后追加 |
| 当前阶段 | `phase()` | Prefill、Decode 或结束事件发生时改变 |
| 失败原因 | `failure_reason()` | 模型执行失败时写入 |
| 教学词元历史 | `history()` | prompt 或生成词元提交后追加 |

`Initial`、`Prefilled` 和 `Decoding` 表示请求仍可推进。`Finished`、`Cancelled`、`TimedOut`、`Rejected` 和 `Failed` 都是终态；终态请求不会再增加 Prefill 游标或生成词元。

Chapter07 命中共享 K/V 前缀时，`adopt_prefilled_prefix()` 会把同一段 prompt 写入词元历史并推进 Prefill 游标。它只接受尚未执行的请求和严格短于完整 prompt 的前缀，防止 Request、模型 Context 与 K/V 块表从不同位置继续执行。

## 配置与模型执行器

生成配置定义停止和准入条件，位于 <a href="/_code/chapter06_generation/include/chapter06_generation/generation_types.h"><code>generation_types.h</code></a>。`max_new_tokens` 限制输出长度，`max_context_tokens` 限制 prompt 与计划输出的总长度，`eos_token_id` 提供提前停止条件。

模型执行器只回答“给定输入后 logits 是什么”，不选择下一个词元，也不改变请求终态。接口位于 <a href="/_code/chapter06_generation/include/chapter06_generation/model_executor.h"><code>model_executor.h</code></a>。

| 调用 | 输入 | 输出 | 模型侧状态 |
| --- | --- | --- | --- |
| `prefill(prompt)` | 完整 prompt | 首个待采样 logits | 创建并填充请求上下文 |
| `prefill_chunk(...)` | 一段 prompt | 最后一段才返回 logits | 逐段增加已提交位置 |
| `decode(token)` | 上一步已生成词元 | 下一组 logits | 追加该词元对应的模型 K/V |
| `execution_state()` | 无 | 已提交词元数和 K/V 层数 | 只读观察，不推进模型 |

可执行程序可以从命令行读取模型路径、后端或实现名称，但字符串解析到此为止。入口把参数转换成 `GenerationConfig`、后端枚举或 Factory 配置，Runtime 和算子只接收类型化对象。这样既能在不改源码的情况下切换实验路径，也能让非法值在模型加载前直接失败。

| 配置层级 | 示例 | 生效时机 |
| --- | --- | --- |
| 构建选项 | `CPP_LLM_ENGINE_ENABLE_CUDA` | CMake 配置时决定是否编译 CUDA |
| 进程实现 | `--attention-path fused-online` | 启动时转换为枚举，交给 Factory/Context |
| 请求配置 | `max_new_tokens`、`eos_token_id` | 每个请求进入 Runtime 时 |
| 实验负载 | `--batch 4`、`--iterations 50` | 只改变 benchmark 输入和采样方式 |

命令行不是新的运行时层。它只是进程入口：`argv -> 类型化配置 -> Factory/Runtime -> Backend`。脚本中的实验编号和结果目录只负责记录证据，不能改变 kernel 实现。

请求归 Generation Core 管理可观察状态；执行器持有真实模型 Context。`history()` 保存词元 ID，用于观察请求推进；真实模型的分层 K/V 由执行器持有，物理 Block 分配在 Chapter07 引入。

## Factory 为什么叫“工厂”

一个真实请求进入 Runtime 时，需要取得已经加载的模型、创建独立 Context，并选择 CPU 或 CUDA 执行器。如果每个调用方分别完成这些步骤，模型可能被重复加载，Context 容量和后端配置也容易不一致。执行器工厂（Executor Factory）把对象创建和共享资源复用集中在一个入口：

```text
执行器工厂（共享模型、后端资源）
  ├─ 创建执行器 A -> Context A -> 请求 A
  └─ 创建执行器 B -> Context B -> 请求 B
```

“Factory”来自工厂模式的命名：调用方只提出“创建一个请求执行器”，工厂决定怎样装配依赖。它不负责请求排队、采样或终态转换。Model 和 Factory 通常在多个请求之间长期复用，Executor 和 Context 随单个请求创建和释放。CUDA 批量临时张量或分页块池也可以由 Factory 复用，但请求的位置和 K/V 绑定仍归各自 Context。

## 验证请求状态

从仓库根目录执行 Request 契约测试：

```bash
cmake -S code/chapter06_generation -B /tmp/cpp-llm-chapter06-contract \
  -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/cpp-llm-chapter06-contract \
  --target chapter06_request_contract_test -j2
ctest --test-dir /tmp/cpp-llm-chapter06-contract \
  -R '^chapter06_request_contract_test$' --output-on-failure
```

通过时可以观察到 prompt 与词元历史分别保存、缓存前缀同步推进两者、Prefill 游标单调递增且不越界、五种终态都拒绝后续状态变化。下一篇使用这组契约区分 Prefill 与 Decode。
