# 06.7 逐词元通知并传播模型错误

同步生成通常在请求结束后返回完整词元序列，流式调用则需要在每个词元提交后立即通知上层。Generation Core 提供单请求词元回调（`TokenCallback`），只传递词元 ID 和请求内 step；HTTP、SSE、断连和写回背压留给 Chapter10。

## 提交后再通知

`Runtime::generate()` 每次成功采样后按固定顺序执行：

1. 把词元追加到 `Request::generated()`；
2. 更新教学词元历史并检查停止条件；
3. 调用 `TokenCallback(token, step)`。

回调触发时，调用者已经能从 Request 读到词元、更新后的历史和当前阶段；EOS 回调会观察到 `Finished`。`step` 从 0 开始，只表示当前请求的输出位置，不包含服务请求 ID，也不表示批次中的行号。回调是同步观察接口；异常会在请求完成本步提交后继续向调用者传播。

| 回调参数 | 含义 | 不包含 |
| --- | --- | --- |
| `token` | 已经提交的输出词元 ID | 文本编码和 HTTP 数据帧 |
| `step` | 当前请求内的输出序号 | Scheduler 顺序和全局序号 |

公共回调类型定义在 <a href="/_code/chapter06_generation/include/chapter06_generation/generation_types.h"><code>generation_types.h</code></a>，单请求分发位于 <a href="/_code/chapter06_generation/src/runtime.cpp"><code>runtime.cpp</code></a>。Chapter08 使用带 `request_id` 的 `TokenStreamCallback` 转发多请求结果，但不改变 Generation 的单请求回调契约。

## 错误怎样结束生成

`RequestModelExecutor` 通过布尔返回值和错误字符串报告 Prefill/Decode 失败。Runtime 只在模型调用成功且 logits 非空时采样；失败时进入 `Failed` 并停止循环。

| 失败位置 | Request 变化 | 回调行为 |
| --- | --- | --- |
| Prefill 返回失败 | `Failed`，保存错误原因 | 不触发 |
| Prefill 返回空 logits | `Failed` | 不触发 |
| Decode 返回失败 | 保留之前已提交词元，随后 `Failed` | 不为失败步骤触发 |
| Decode 返回空 logits | 保留之前已提交词元，随后 `Failed` | 不为失败步骤触发 |

当前 `TokenCallback` 是同步观察接口，不返回写入结果。网络断连、慢客户端和 Writer 失败需要由 Chapter10 把请求标记为取消或失败，不能在 Generation Core 中引入 socket 状态。

## 验证回调时序

```bash
cmake -S code/chapter06_generation -B /tmp/cpp-llm-chapter06-stream \
  -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/cpp-llm-chapter06-stream \
  --target chapter06_generation_test -j2
ctest --test-dir /tmp/cpp-llm-chapter06-stream \
  -R '^chapter06_generation_test$' --output-on-failure
```

Generation 测试固定 EOS 词元 `7`，检查回调收到 `(7, 0)`，并观察到输出长度 `1`、历史长度 `3` 和 `Finished`。多请求回调归属和 step 顺序由 Chapter08 的 Scheduler 测试验证。
