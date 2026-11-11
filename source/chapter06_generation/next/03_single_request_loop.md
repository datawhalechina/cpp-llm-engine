# 06.3 运行单请求逐词元生成循环

请求状态规则和 Prefill/Decode 顺序确定后，可以把一次模型前向扩展为持续生成。循环只推进一个请求，不选择多请求执行顺序，也不分配物理 KV Block；它提供后续缓存和调度都必须保持的结果基线。

## 从首个 logits 到终态

下面的算法镜像对应 `Runtime::generate(Request&, Sampler&, RequestModelExecutor&)`。它省略 C++ 错误字符串和所有权，只保留可以逐步核对的控制流。

```python
def generate_one(request, executor, sampler, config):
    logits = executor.prefill(request.prompt)
    request.mark_prefilled()

    while not request.terminal():
        token = sampler.sample(logits)
        request.append_generated(token)

        if token == config.eos_token_id:
            request.mark_finished()
        elif request.generated_size == config.max_new_tokens:
            request.mark_finished()
        else:
            logits = executor.decode(token)

    return request.generated
```

真实 C++ 路径把第一次 Prefill 和采样封装在 `prefill_chunk()`，后续循环调用 `decode_one()`。实现位于 <a href="https://github.com/datawhalechina/cpp-llm-engine/blob/main/code/chapter06_generation/src/runtime.cpp"><code>runtime.cpp</code></a>，确定性采样位于 <a href="https://github.com/datawhalechina/cpp-llm-engine/blob/main/code/chapter06_generation/src/sampler.cpp"><code>sampler.cpp</code></a>。

## 每一步检查什么

| 检查点 | 成功条件 | 失败后的状态 |
| --- | --- | --- |
| Runtime 可运行 | `max_new_tokens > 0` | 不调用模型 |
| 上下文准入 | prompt 与计划输出不超过限制 | `Rejected` |
| Prefill | 执行器成功且最终 logits 非空 | `Failed` |
| 采样 | logits 能产生词元 ID | 当前实现使用 Greedy 基线 |
| Decode | 执行器成功且 logits 非空 | `Failed` |
| 停止 | 命中 EOS 或输出长度上限 | `Finished` |

Greedy Sampler 选择最大 logits 的索引，同一组 logits 总是得到相同词元。采样策略与模型执行分离后，可以单独验证停止条件，也可以在不修改 Runtime 的情况下增加 temperature、top-k 或 top-p。

单请求循环只消费请求自己的模型 Context。多个请求即使共享只读模型权重，也必须分别持有位置和 K/V；把哪些请求放入同一步执行由 Chapter08 决定。

## 运行最小闭环

构建并运行单请求示例：

```bash
cmake -S code/chapter06_generation -B /tmp/cpp-llm-chapter06-loop \
  -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/cpp-llm-chapter06-loop \
  --target chapter06_generation_demo -j2
/tmp/cpp-llm-chapter06-loop/chapter06_generation_demo
```

输出应包含 `prefills: 3`、`generated steps: 2`、`token history size: 5` 和 `request phase: finished`。随后执行：

```bash
ctest --test-dir /tmp/cpp-llm-chapter06-loop \
  -R 'chapter06_(generation|sampler)_test' --output-on-failure
```

测试通过后，单请求已经具有确定的输入、输出和停止语义。接下来依次核对状态机、采样与停止、逐词元通知和真实模型入口，确认这些扩展继续保持同一组请求推进规则。
