// Chapter 07：实现 benchmark 检查和 trace/summary 文本、JSON 渲染。
// 职责：把固定阶段数据转换成终端输出、JSON 和 Chrome trace 格式。
// 验证：Chapter 07 测试检查 unit/e2e 状态和关键输出字段。
#include "chapter07_benchmark/test_helpers.h"

#include "chapter07_benchmark/benchmark.h"

#include <iomanip>
#include <sstream>

namespace chapter07_benchmark {

namespace {

int trace_tid_for_phase(const std::string& phase) {
    if (phase == "prefill" || phase == "decode") {
        return 1;
    }
    return 0;
}

const char* trace_track_name(int tid) {
    return tid == 1 ? "compute-stream" : "host-thread";
}

}  // namespace

bool run_unit_checks() {
    const BenchmarkResult result = make_result("unit", 10, 2.0);
    return result.name == "unit" && result.iterations == 10 && result.items_per_second > 0.0;
}

bool run_e2e_checks() {
    const auto samples = sample_generation_path();
    return samples.size() == 6 && samples.front().phase == "input" && samples.back().phase == "benchmark";
}

std::string render_summary() {
    const auto samples = sample_generation_path();
    double total_ms = 0.0;
    for (const auto& sample : samples) {
        if (sample.phase == "benchmark") {
            total_ms = sample.elapsed_ms;
            break;
        }
    }

    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2);
    oss << "chapter07 benchmark summary\n";
    for (const auto& sample : samples) {
        oss << sample.phase << ": tokens=" << sample.tokens << ", elapsed_ms=" << sample.elapsed_ms;
        if (sample.phase != "benchmark") {
            oss << ", share=" << phase_share(sample.elapsed_ms, total_ms) << "%";
            if (sample.tokens > 0) {
                oss << ", tokens_per_second=" << tokens_per_second(sample.tokens, sample.elapsed_ms);
            }
        }
        oss << '\n';
    }
    return oss.str();
}

std::string render_trace() {
    const auto events = sample_trace_events();
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2);
    oss << "chapter07 trace events\n";
    for (const auto& event : events) {
        oss << event.phase
            << " | " << event.event
            << " | start_ms=" << event.start_ms
            << " | duration_ms=" << event.duration_ms
            << " | note=" << event.note
            << '\n';
    }
    return oss.str();
}

std::string render_hotspots() {
    const auto hotspots = sample_hotspots();
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2);
    oss << "chapter07 hotspot summary\n";
    for (const auto& entry : hotspots) {
        oss << entry.phase
            << " | " << entry.symbol
            << " | self_ms=" << entry.self_ms
            << " | total_ms=" << entry.total_ms
            << " | calls=" << entry.calls
            << " | reason=" << entry.reason
            << '\n';
    }
    return oss.str();
}

std::string render_trace_json() {
    const auto samples = sample_generation_path();
    const auto events = sample_trace_events();
    const auto hotspots = sample_hotspots();

    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2);
    oss << "{\n";
    oss << "  \"benchmark\": {\n";
    oss << "    \"name\": \"chapter07_demo\",\n";
    oss << "    \"items_per_second\": 4000.00,\n";
    oss << "    \"phases\": [\n";
    for (std::size_t i = 0; i < samples.size(); ++i) {
        const auto& sample = samples[i];
        oss << "      {\"phase\": \"" << sample.phase << "\", \"tokens\": " << sample.tokens
            << ", \"elapsed_ms\": " << sample.elapsed_ms << "}";
        oss << (i + 1 == samples.size() ? "\n" : ",\n");
    }
    oss << "    ]\n";
    oss << "  },\n";
    oss << "  \"trace_events\": [\n";
    for (std::size_t i = 0; i < events.size(); ++i) {
        const auto& event = events[i];
        oss << "    {\"phase\": \"" << event.phase
            << "\", \"event\": \"" << event.event
            << "\", \"start_ms\": " << event.start_ms
            << ", \"duration_ms\": " << event.duration_ms
            << ", \"note\": \"" << event.note << "\"}";
        oss << (i + 1 == events.size() ? "\n" : ",\n");
    }
    oss << "  ],\n";
    oss << "  \"hotspots\": [\n";
    for (std::size_t i = 0; i < hotspots.size(); ++i) {
        const auto& hotspot = hotspots[i];
        oss << "    {\"phase\": \"" << hotspot.phase
            << "\", \"symbol\": \"" << hotspot.symbol
            << "\", \"self_ms\": " << hotspot.self_ms
            << ", \"total_ms\": " << hotspot.total_ms
            << ", \"calls\": " << hotspot.calls
            << ", \"reason\": \"" << hotspot.reason << "\"}";
        oss << (i + 1 == hotspots.size() ? "\n" : ",\n");
    }
    oss << "  ]\n";
    oss << "}\n";
    return oss.str();
}

std::string render_chrome_trace_json() {
    const auto events = sample_trace_events();

    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2);
    oss << "{\n";
    oss << "  \"traceEvents\": [\n";
    oss << "    {\"name\": \"thread_name\", \"ph\": \"M\", \"pid\": 1, \"tid\": 0, \"args\": {\"name\": \"host-thread\"}},\n";
    oss << "    {\"name\": \"thread_name\", \"ph\": \"M\", \"pid\": 1, \"tid\": 1, \"args\": {\"name\": \"compute-stream\"}},\n";
    for (std::size_t i = 0; i < events.size(); ++i) {
        const auto& event = events[i];
        const int tid = trace_tid_for_phase(event.phase);
        oss << "    {\"name\": \"" << event.event
            << "\", \"cat\": \"" << event.phase
            << "\", \"ph\": \"X\", \"pid\": 1, \"tid\": " << tid
            << ", \"ts\": " << (event.start_ms * 1000.0)
            << ", \"dur\": " << (event.duration_ms * 1000.0)
            << ", \"args\": {\"note\": \"" << event.note
            << "\", \"track\": \"" << trace_track_name(tid) << "\"}}";
        oss << ",\n";
        if (event.event == "sample_logits") {
            oss << "    {\"name\": \"sync_decode_result\", \"cat\": \"sync\", \"ph\": \"i\", \"s\": \"t\", \"pid\": 1, \"tid\": 0"
                << ", \"ts\": 450.00, \"args\": {\"note\": \"host waits for decode result\"}},\n";
        }
    }
    oss << "    {\"name\": \"trace_end\", \"ph\": \"i\", \"s\": \"g\", \"pid\": 1, \"tid\": 0, \"ts\": 600.00, \"args\": {\"note\": \"demo boundary\"}}\n";
    oss << "  ]\n";
    oss << "}\n";
    return oss.str();
}

std::string render_multi_request_trace_json() {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2);
    oss << "{\n";
    oss << "  \"traceEvents\": [\n";
    oss << "    {\"name\": \"thread_name\", \"ph\": \"M\", \"pid\": 1, \"tid\": 0, \"args\": {\"name\": \"host-thread\"}},\n";
    oss << "    {\"name\": \"thread_name\", \"ph\": \"M\", \"pid\": 1, \"tid\": 1, \"args\": {\"name\": \"compute-stream-0\"}},\n";
    oss << "    {\"name\": \"thread_name\", \"ph\": \"M\", \"pid\": 1, \"tid\": 2, \"args\": {\"name\": \"compute-stream-1\"}},\n";
    oss << "    {\"name\": \"req-A prefill\", \"cat\": \"prefill\", \"ph\": \"X\", \"pid\": 1, \"tid\": 1, \"ts\": 0.00, \"dur\": 180.00, \"args\": {\"request\": \"A\"}},\n";
    oss << "    {\"name\": \"req-B admit\", \"cat\": \"scheduler\", \"ph\": \"i\", \"s\": \"t\", \"pid\": 1, \"tid\": 0, \"ts\": 70.00, \"args\": {\"request\": \"B\", \"note\": \"request enters while A is still running\"}},\n";
    oss << "    {\"name\": \"req-A decode\", \"cat\": \"decode\", \"ph\": \"X\", \"pid\": 1, \"tid\": 1, \"ts\": 180.00, \"dur\": 90.00, \"args\": {\"request\": \"A\"}},\n";
    oss << "    {\"name\": \"req-B prefill\", \"cat\": \"prefill\", \"ph\": \"X\", \"pid\": 1, \"tid\": 2, \"ts\": 110.00, \"dur\": 160.00, \"args\": {\"request\": \"B\"}},\n";
    oss << "    {\"name\": \"sync_batch_merge\", \"cat\": \"sync\", \"ph\": \"i\", \"s\": \"t\", \"pid\": 1, \"tid\": 0, \"ts\": 270.00, \"args\": {\"note\": \"host merges ready decode work\"}},\n";
    oss << "    {\"name\": \"req-A stream\", \"cat\": \"output\", \"ph\": \"X\", \"pid\": 1, \"tid\": 0, \"ts\": 275.00, \"dur\": 40.00, \"args\": {\"request\": \"A\"}},\n";
    oss << "    {\"name\": \"req-B decode\", \"cat\": \"decode\", \"ph\": \"X\", \"pid\": 1, \"tid\": 2, \"ts\": 270.00, \"dur\": 100.00, \"args\": {\"request\": \"B\"}},\n";
    oss << "    {\"name\": \"req-B stream\", \"cat\": \"output\", \"ph\": \"X\", \"pid\": 1, \"tid\": 0, \"ts\": 370.00, \"dur\": 35.00, \"args\": {\"request\": \"B\"}},\n";
    oss << "    {\"name\": \"trace_end\", \"ph\": \"i\", \"s\": \"g\", \"pid\": 1, \"tid\": 0, \"ts\": 405.00, \"args\": {\"note\": \"multi-request demo boundary\"}}\n";
    oss << "  ]\n";
    oss << "}\n";
    return oss.str();
}

}  // namespace chapter07_benchmark
