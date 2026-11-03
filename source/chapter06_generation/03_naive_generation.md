# 06.3 建立逐词元生成循环

多层模型已经能从 token 序列产生 logits，现在需要把一次前向变成持续生成。最小闭环由 `Runtime`、`RequestModelExecutor`、`Sampler` 和 `Request` 协作完成；先固定结果语义，后续 KV Cache 与 Scheduler 才有可比较的基线。

## 逐 Token 生成循环

下面的算法镜像只保留生成控制流，省略 C++ 所有权、错误状态、streaming callback 和真实模型 Context。仓库入口是 <a href="../_code/chapter06_generation/src/runtime.cpp"><code>runtime.cpp</code></a>，请求状态定义在 <a href="../_code/chapter06_generation/src/request.cpp"><code>request.cpp</code></a>。

```python
def generate(request, model_executor, sampler):
    logits = model_executor.prefill(request.prompt)
    request.mark_prefilled()

    while not request.terminal():
        next_token = sampler.sample(logits)
        request.append_generated(next_token)

        if next_token == request.eos_token_id:
            request.finish("eos")
        elif request.generated_size == request.max_new_tokens:
            request.finish("token_budget")
        elif request.context_size == request.max_context_length:
            request.finish("context_limit")
        else:
            logits = model_executor.decode(next_token)

    return request.generated_tokens
```

| 算法动作 | C++ 对象或源码 | 验证重点 |
| --- | --- | --- |
| prompt 建立初始状态 | `Runtime::prefill()` | prefill 只发生一次，首个 logits 有效 |
| logits 选择 token | `Sampler::sample()` | 固定 logits 得到固定 token |
| token 与状态提交 | `Request::append_generated()` | history、预算和终态同步更新 |
| 增量推进模型 | `Runtime::decode_one()` | 每次只消费所属请求的 Context |

## 完整重算与增量执行

完整重算会在每一步重新处理全部历史 token。prompt 长度为 `S`、生成 `T` 个 token 时，Attention 相关工作近似累积为：

```text
S^2 + (S + 1)^2 + ... + (S + T - 1)^2
```

真实 `RealLlamaCpuContext` 在 prefill 后保存每层 K/V，decode 只计算新 token 并读取历史缓存。Chapter06 的教学 `KVCache` 保存 token ID history，用于观察请求状态；它与模型 Context 内的分层 K/V 是不同对象。

| 对象 | 在生成闭环中的职责 |
| --- | --- |
| `Request` | 保存 prompt、生成 token、状态和停止原因 |
| `Runtime` | 执行 prefill/decode，追加结果并推进状态 |
| `RequestModelExecutor` | 持有单请求 Model Context，返回 logits |
| `Sampler` | 从 logits 选择下一个 token |
| `Scheduler` | 决定哪个请求在当前 step 被推进 |

朴素路径提供可读、可重复的正确性基线。引入缓存或批处理后仍应使用相同 prompt、Sampler 和停止条件比较输出，避免把语义变化误判为性能收益。

## 验证与衔接

```bash
bash code/chapter06_generation/scripts/run.sh
```

检查单请求可以连续生成，EOS、token budget 和 context limit 都能结束循环；相同输入和 Greedy Sampler 得到固定输出；模型失败不会产生占位 token。随后阅读 [`06.4 KV Cache`](./04_kv_cache.md)，分别跟踪 token history、模型分层 K/V 和 block ownership。
