// Chapter 07：实现 benchmark 结果计算、阶段吞吐和 hotspot 数据样本。
// 职责：提供不依赖计时器的纯计算基线，供 demo、测试和文档示例复用。
// 验证：chapter07_benchmark_test 检查边界输入和固定样本结果。
#include "chapter07_benchmark/benchmark.h"

#include <numeric>

namespace chapter07_benchmark {

BenchmarkResult make_result(const std::string& name, std::size_t iterations, double elapsed_ms) {
    BenchmarkResult result;
    result.name = name;
    result.iterations = iterations;
    result.elapsed_ms = elapsed_ms;
    result.items_per_second = iterations > 0 && elapsed_ms > 0.0
                                  ? (static_cast<double>(iterations) * 1000.0 / elapsed_ms)
                                  : 0.0;
    return result;
}

double tokens_per_second(std::size_t tokens, double elapsed_ms) {
    if (tokens == 0 || elapsed_ms <= 0.0) {
        return 0.0;
    }
    return static_cast<double>(tokens) * 1000.0 / elapsed_ms;
}

double phase_share(double elapsed_ms, double total_ms) {
    if (elapsed_ms < 0.0 || total_ms <= 0.0) {
        return 0.0;
    }
    return elapsed_ms * 100.0 / total_ms;
}

std::vector<BenchmarkSample> sample_generation_path() {
    return {
        {"input", 5, 0.05},
        {"prefill", 5, 0.30},
        {"decode", 2, 0.15},
        {"sampling", 2, 0.05},
        {"output", 2, 0.05},
        {"benchmark", 7, 0.60},
    };
}

std::vector<TraceEvent> sample_trace_events() {
    return {
        {"input", "prepare_tokens", 0.00, 0.03, "token ids and positions"},
        {"input", "copy_request_state", 0.03, 0.02, "request metadata"},
        {"prefill", "embed_prompt", 0.05, 0.08, "prompt embedding"},
        {"prefill", "run_prefill_attention", 0.13, 0.14, "largest prefill slice"},
        {"prefill", "write_kv_cache", 0.27, 0.03, "initial cache materialization"},
        {"decode", "decode_step", 0.30, 0.09, "single token decode loop"},
        {"decode", "sample_logits", 0.39, 0.06, "logits to next token"},
        {"sampling", "topk_filter", 0.45, 0.03, "sampling filter"},
        {"sampling", "commit_token", 0.48, 0.02, "append generated token"},
        {"output", "format_response", 0.50, 0.03, "prepare response payload"},
        {"output", "flush_stream", 0.53, 0.05, "emit partial output"},
        {"output", "finalize_request", 0.58, 0.02, "request teardown"},
    };
}

std::vector<HotspotEntry> sample_hotspots() {
    return {
        {"prefill", "run_prefill_attention", 0.14, 0.17, 1, "largest prefill compute slice"},
        {"decode", "decode_step", 0.09, 0.15, 1, "single token decode loop"},
        {"prefill", "embed_prompt", 0.08, 0.08, 1, "prompt embedding path"},
        {"decode", "sample_logits", 0.06, 0.06, 1, "decode tail after model step"},
        {"output", "flush_stream", 0.05, 0.05, 1, "response emission dominates output"},
    };
}

}  // namespace chapter07_benchmark
