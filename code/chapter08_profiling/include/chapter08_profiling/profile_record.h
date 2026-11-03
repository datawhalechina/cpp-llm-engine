// Chapter 08：声明教学与真实服务 benchmark 共用的 profiling 记录。
// 职责：保存 workload、环境、正确性、阶段耗时和多请求延迟差异。
// 验证：Chapter08 CTest 与 profiling 脚本检查解析、报告和 A/B 可比性。
#pragma once

#include <cstddef>
#include <iosfwd>
#include <string>

namespace chapter08_profiling {

struct ProfileRecord {
    std::string experiment_id = "unregistered";
    std::string workload_id = "unknown";
    std::string backend = "unknown";
    std::string model = "unknown";
    std::string cpu_model = "unknown";
    std::string build_type = "unknown";
    std::string execution_pattern = "unknown";
    bool correctness_pass = false;
    std::size_t iterations = 0;
    std::size_t request_count = 0;
    std::size_t generated_tokens_per_iteration = 0;
    double prefill_ms = 0.0;
    double decode_ms = 0.0;
    double runtime_overhead_ms = 0.0;
    double total_ms = 0.0;
    double ttft_ms = 0.0;
    double completion_ms = 0.0;
    double ttft_spread_ms = 0.0;
    double completion_spread_ms = 0.0;
    double reported_tokens_per_second = 0.0;
    std::size_t model_weight_payload_bytes = 0;
    std::size_t kv_cache_bytes = 0;
    double peak_memory_mb = 0.0;

    double tokens_per_second() const;
    double model_compute_share_percent() const;
    double runtime_overhead_share_percent() const;
    bool valid(std::string* error = nullptr) const;
};

ProfileRecord parse_benchmark_output(std::istream& input);
std::string render_report(const ProfileRecord& record);

}  // namespace chapter08_profiling
