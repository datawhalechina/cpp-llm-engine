// Chapter 07：声明 unit/e2e 检查和文本、JSON、Chrome trace 渲染辅助函数。
// 职责：把固定 benchmark 样本转换为学习者可读或工具可消费的结果。
// 验证：Chapter 07 demo 和测试检查各类输出包含预期字段。
#pragma once

#include <string>

namespace chapter07_benchmark {

bool run_unit_checks();
bool run_e2e_checks();
std::string render_summary();
std::string render_trace();
std::string render_hotspots();
std::string render_trace_json();
std::string render_chrome_trace_json();
std::string render_multi_request_trace_json();

}  // namespace chapter07_benchmark
