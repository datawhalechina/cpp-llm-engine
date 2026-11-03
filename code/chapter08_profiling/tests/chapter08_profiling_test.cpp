// Chapter 08：验证教学与真实服务 profiling 字段、报告和 A/B 判定。
// 契约：correctness 与 workload 可比性先于性能 delta。
// 验证入口：Chapter08 CTest 或 scripts/run.sh。
#include "chapter08_profiling/ab_compare.h"

#include <cmath>
#include <iostream>
#include <sstream>

namespace {

using chapter08_profiling::ProfileRecord;

ProfileRecord valid_record(double total_ms) {
    ProfileRecord record;
    record.experiment_id = "EXP-TEST-01";
    record.workload_id = "WL-TEST-CPU-V1";
    record.backend = "cpu";
    record.correctness_pass = true;
    record.iterations = 3;
    record.generated_tokens_per_iteration = 4;
    record.prefill_ms = total_ms * 0.4;
    record.decode_ms = total_ms * 0.6;
    record.total_ms = total_ms;
    return record;
}

bool test_parse() {
    std::istringstream input(
        "experiment_id=EXP-CPU-01\n"
        "workload_id=WL-CPU-TEACHING-BATCH-V1\n"
        "backend=cpu\n"
        "correctness=PASS\n"
        "iterations=3\n"
        "generated_tokens_per_iteration=4\n"
        "prefill_ms=1.5\n"
        "decode_ms=2.5\n"
        "ttft_ms=1.8\n"
        "model_weight_payload_bytes=667078656\n"
        "kv_cache_bytes=585728\n"
        "peak_memory_mb=652.8203\n");
    const ProfileRecord record = chapter08_profiling::parse_benchmark_output(input);
    const std::string report = chapter08_profiling::render_report(record);
    return record.experiment_id == "EXP-CPU-01" &&
           record.workload_id == "WL-CPU-TEACHING-BATCH-V1" &&
           record.backend == "cpu" && record.total_ms == 4.0 &&
           record.ttft_ms == 1.8 &&
           record.model_weight_payload_bytes == 667078656 &&
           record.kv_cache_bytes == 585728 &&
           std::abs(record.peak_memory_mb - 652.8203) < 1e-6 &&
           report.find("model_weight_payload_bytes=667078656") != std::string::npos &&
           report.find("kv_cache_bytes=585728") != std::string::npos &&
           record.valid();
}

bool test_tokens_per_second() {
    const ProfileRecord record = valid_record(2.0);
    return std::abs(record.tokens_per_second() - 2000.0) < 1e-6;
}

bool test_accepts_small_regression() {
    const auto result = chapter08_profiling::compare(valid_record(10.0), valid_record(10.4));
    return result.accepted && result.correctness_ok;
}

bool test_rejects_large_regression() {
    const auto result = chapter08_profiling::compare(valid_record(10.0), valid_record(11.0));
    return !result.accepted && result.reason.find("regression") != std::string::npos;
}

bool test_rejects_failed_correctness() {
    ProfileRecord experiment = valid_record(1.0);
    experiment.correctness_pass = false;
    const auto result = chapter08_profiling::compare(valid_record(10.0), experiment);
    return !result.accepted && !result.correctness_ok;
}

bool test_reports_dominant_phase() {
    ProfileRecord record = valid_record(10.0);
    record.prefill_ms = 3.96;
    record.decode_ms = 5.94;
    record.runtime_overhead_ms = 0.1;
    record.execution_pattern = "single_thread_round_robin";
    const std::string report = chapter08_profiling::render_report(record);
    return report.find("dominant_phase=decode") != std::string::npos &&
           report.find("prefill_share_percent=40.0000") != std::string::npos &&
           report.find("model_compute_share_percent=99.0000") != std::string::npos &&
           report.find("runtime_overhead_share_percent=1.0000") != std::string::npos &&
           report.find("bottleneck_scope=model_compute") != std::string::npos;
}

bool test_rejects_incomparable_workload() {
    ProfileRecord experiment = valid_record(9.0);
    experiment.workload_id = "WL-OTHER-V1";
    const auto result = chapter08_profiling::compare(valid_record(10.0), experiment);
    return !result.accepted && !result.correctness_ok &&
           result.reason.find("not comparable") != std::string::npos;
}

}  // namespace

int main() {
    const bool passed = test_parse() && test_tokens_per_second() &&
                        test_accepts_small_regression() && test_rejects_large_regression() &&
                        test_rejects_failed_correctness() && test_reports_dominant_phase() &&
                        test_rejects_incomparable_workload();
    std::cout << "chapter08 profiling tests: " << (passed ? "PASS" : "FAIL") << '\n';
    return passed ? 0 : 1;
}
