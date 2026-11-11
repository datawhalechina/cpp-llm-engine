# 10.5 完成一次同步 Completion 请求

同步 Completion 在响应前跑完请求，适合验证最短服务闭环。应用层先把文本转换为 token ID，再由 <a href="https://github.com/datawhalechina/cpp-llm-engine/blob/main/code/chapter10_serving/src/http_adapter.cpp"><code>HttpAdapter</code></a> 创建 Request 和请求专属 Executor，ServiceCore 完成准入、执行与释放。

## 请求路径

| 阶段 | 主要动作 | 失败映射 |
| --- | --- | --- |
| 解析 | method、path、request ID、prompt tokens | `400` 或 `404` |
| 创建 | Request、Sampler 引用、Executor/Context | `500` |
| 准入 | 重复 ID 和容量检查 | `409` 或 `429` |
| 执行 | 调度 step 直到终态 | `408`、`499` 或 `500` |
| 响应 | token IDs、数量、finish reason | `200` |
| 释放 | K/V、Context 和服务条目 | 指标与 inflight 核对 |

## Tokenizer 放在应用组合层

Transport 不认识模型词表，ServiceCore 也不重复实现分词。文本 API 由应用持有真实 Tokenizer，把 token 序列交给稳定的服务契约；教学测试可以直接使用 token ID，隔离模型文件依赖。

| 输入形式 | 适用场景 | 谁完成转换 |
| --- | --- | --- |
| `prompt_tokens` | 确定性单元与契约测试 | 测试直接构造 |
| 文本 `prompt` | 真实模型服务 | 应用层真实 Tokenizer |

## 验证同步请求

运行 `bash code/chapter10_serving/scripts/run.sh`。`chapter10_http_contract_test` 应检查路由、参数、准入失败和成功 Completion 的状态码与响应字段，并确认请求终态后资源已释放。`10.6` 把同一生命周期改为逐词元写回。
