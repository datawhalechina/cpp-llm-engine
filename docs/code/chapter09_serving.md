# Chapter 09：Serving Extensions

代码目录：<a href="/_code/chapter09_serving/README.md"><code>code/chapter09_serving</code></a>

本章实现最小 C++ Service Core，覆盖 admission、waiting/running 队列、batch 上限、取消/超时、在线延迟指标和完成后 KV Cache/Executor 回收。可选真实模型测试由 `code/engine` 提供 Factory，为同步和 SSE 请求分别创建独立 Context。

正文从 [`Chapter 09 导读`](/chapter09_serving/intro) 进入；网络库评估和 transport 接入见 [`Chapter 09 Transport`](/chapter09_serving/02_transport)。

## 资产

- <a href="/_code/chapter09_serving/include/chapter09_serving/serving_core.h"><code>serving_core.h</code></a>
- <a href="/_code/chapter09_serving/src/serving_core.cpp"><code>serving_core.cpp</code></a>
- <a href="/_code/chapter09_serving/examples/main_serving.cpp"><code>main_serving.cpp</code></a>
- <a href="/_code/chapter09_serving/tests/chapter09_serving_test.cpp"><code>chapter09_serving_test.cpp</code></a>
- <a href="/_code/chapter09_serving/tests/step_dispatch_test.cpp"><code>step_dispatch_test.cpp</code></a>
- <a href="/_code/chapter09_serving/tests/parallel_dispatch_test.cpp"><code>parallel_dispatch_test.cpp</code></a>
- <a href="/_code/chapter09_serving/tests/cancellation_release_test.cpp"><code>cancellation_release_test.cpp</code></a>
- <a href="/_code/chapter09_serving/tests/service_metrics_test.cpp"><code>service_metrics_test.cpp</code></a>
- <a href="/_code/chapter09_serving/tests/fault_sequence_test.cpp"><code>fault_sequence_test.cpp</code></a>
- <a href="/_code/chapter09_serving/include/chapter09_serving/response_writer.h"><code>response_writer.h</code></a>
- <a href="/_code/chapter09_serving/include/chapter09_serving/cpp_httplib_transport.h"><code>cpp_httplib_transport.h</code></a>
- <a href="/_code/chapter09_serving/src/cpp_httplib_transport.cpp"><code>cpp_httplib_transport.cpp</code></a>
- <a href="/_code/chapter09_serving/scripts/run.sh"><code>run.sh</code></a>
- <a href="/_code/chapter09_serving/scripts/run_cpp_httplib_loopback.sh"><code>run_cpp_httplib_loopback.sh</code></a>
- <a href="/_code/engine/include/cpp_llm_engine/real_llama_executor_factory.h"><code>real_llama_executor_factory.h</code></a>
- <a href="/_code/engine/examples/real_tinyllama_serving_smoke.cpp"><code>real_tinyllama_serving_smoke.cpp</code></a>
- <a href="/_code/engine/examples/real_tinyllama_http_adapter_smoke.cpp"><code>real_tinyllama_http_adapter_smoke.cpp</code></a>
- <a href="/_code/engine/examples/real_tinyllama_serving_benchmark.cpp"><code>real_tinyllama_serving_benchmark.cpp</code></a>
- <a href="/_code/chapter09_serving/results/tinyllama_service_cpu_baseline.json"><code>tinyllama_service_cpu_baseline.json</code></a>

## 验证

```bash
bash code/chapter09_serving/scripts/run.sh
```
