// Chapter 07：测试结果计算、阶段指标、trace 和 hotspot 输出。
// 职责：验证 benchmark 数据结构和渲染接口的正确性，不测量机器性能。
// 验证：由 CTest 和 code/chapter07_benchmark/scripts/run.sh 执行。
#include "chapter07_benchmark/benchmark.h"
#include "chapter07_benchmark/test_framework.h"
#include "chapter07_benchmark/test_helpers.h"

#include <cmath>
#include <string>

namespace {

using chapter07_benchmark::BenchmarkResult;

bool test_make_result() {
    const BenchmarkResult result = chapter07_benchmark::make_result("demo", 100, 25.0);
    return result.name == "demo" &&
           result.iterations == 100 &&
           std::abs(result.items_per_second - 4000.0) < 1e-5;
}

bool test_tokens_per_second() {
    return std::abs(chapter07_benchmark::tokens_per_second(10, 2.0) - 5000.0) < 1e-5;
}

bool test_phase_share() {
    return std::abs(chapter07_benchmark::phase_share(0.3, 0.6) - 50.0) < 1e-5;
}

bool test_unit_checks() {
    return chapter07_benchmark::run_unit_checks();
}

bool test_e2e_checks() {
    return chapter07_benchmark::run_e2e_checks();
}

bool test_summary_text() {
    return chapter07_benchmark::render_summary().find("prefill") != std::string::npos;
}

bool test_trace_text() {
    const std::string trace = chapter07_benchmark::render_trace();
    return trace.find("run_prefill_attention") != std::string::npos &&
           trace.find("flush_stream") != std::string::npos;
}

bool test_hotspot_text() {
    const std::string hotspots = chapter07_benchmark::render_hotspots();
    return hotspots.find("chapter07 hotspot summary") != std::string::npos &&
           hotspots.find("decode_step") != std::string::npos;
}

bool test_trace_json_text() {
    const std::string json = chapter07_benchmark::render_trace_json();
    return json.find("\"trace_events\"") != std::string::npos &&
           json.find("\"hotspots\"") != std::string::npos &&
           json.find("\"run_prefill_attention\"") != std::string::npos;
}

bool test_chrome_trace_json_text() {
    const std::string json = chapter07_benchmark::render_chrome_trace_json();
    return json.find("\"traceEvents\"") != std::string::npos &&
           json.find("\"ph\": \"X\"") != std::string::npos &&
           json.find("\"name\": \"decode_step\"") != std::string::npos &&
           json.find("\"tid\": 1") != std::string::npos &&
           json.find("\"name\": \"sync_decode_result\"") != std::string::npos;
}

bool test_multi_request_trace_json_text() {
    const std::string json = chapter07_benchmark::render_multi_request_trace_json();
    return json.find("\"name\": \"req-A prefill\"") != std::string::npos &&
           json.find("\"name\": \"req-B prefill\"") != std::string::npos &&
           json.find("\"name\": \"sync_batch_merge\"") != std::string::npos &&
           json.find("\"tid\": 2") != std::string::npos;
}

}  // namespace

int main() {
    const auto summary = chapter07_benchmark::run_test_suite({
        {"make_result", &test_make_result},
        {"tokens_per_second", &test_tokens_per_second},
        {"phase_share", &test_phase_share},
        {"unit_checks", &test_unit_checks},
        {"e2e_checks", &test_e2e_checks},
        {"summary_text", &test_summary_text},
        {"trace_text", &test_trace_text},
        {"hotspot_text", &test_hotspot_text},
        {"trace_json_text", &test_trace_json_text},
        {"chrome_trace_json_text", &test_chrome_trace_json_text},
        {"multi_request_trace_json_text", &test_multi_request_trace_json_text},
    });

    return summary.failed == 0 ? 0 : 1;
}
