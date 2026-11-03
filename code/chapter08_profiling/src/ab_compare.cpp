// Chapter 08：实现 profiling A/B 的可比性和延迟回退规则。
// 契约：正确性与 workload 条件是硬闸门，总延迟回退阈值决定 KEEP/REJECT。
// 验证：测试覆盖阶段 delta、吞吐 delta 和不可比输入。
#include "chapter08_profiling/ab_compare.h"

#include <cmath>
#include <iomanip>
#include <sstream>

namespace chapter08_profiling {

namespace {

double delta_percent(double baseline, double experiment) {
    return baseline > 0.0 ? (experiment - baseline) * 100.0 / baseline : 0.0;
}

bool optional_field_matches(const std::string& baseline, const std::string& experiment) {
    return baseline == "unknown" || experiment == "unknown" || baseline == experiment;
}

}  // namespace

ComparisonResult compare(const ProfileRecord& baseline,
                         const ProfileRecord& experiment,
                         ComparisonPolicy policy) {
    ComparisonResult result;
    std::string error;
    if (!baseline.valid(&error)) {
        result.reason = "invalid baseline: " + error;
        return result;
    }
    if (!experiment.valid(&error)) {
        result.reason = "invalid experiment: " + error;
        return result;
    }
    if (baseline.workload_id != experiment.workload_id ||
        baseline.backend != experiment.backend ||
        baseline.generated_tokens_per_iteration !=
            experiment.generated_tokens_per_iteration ||
        (baseline.request_count > 0 && experiment.request_count > 0 &&
         baseline.request_count != experiment.request_count) ||
        !optional_field_matches(baseline.model, experiment.model) ||
        !optional_field_matches(baseline.cpu_model, experiment.cpu_model) ||
        !optional_field_matches(baseline.build_type, experiment.build_type)) {
        result.reason = "baseline and experiment workloads are not comparable";
        return result;
    }

    result.correctness_ok = true;
    result.latency_delta_percent = delta_percent(baseline.total_ms, experiment.total_ms);
    result.prefill_delta_percent = delta_percent(baseline.prefill_ms, experiment.prefill_ms);
    result.decode_delta_percent = delta_percent(baseline.decode_ms, experiment.decode_ms);
    result.throughput_delta_percent = delta_percent(
        baseline.tokens_per_second(), experiment.tokens_per_second());
    result.accepted = result.latency_delta_percent <= policy.max_latency_regression_percent;
    result.reason = result.accepted ? "accepted" : "latency regression exceeds policy";
    return result;
}

std::string render_comparison(const ComparisonResult& result) {
    std::ostringstream output;
    output << std::fixed << std::setprecision(4)
           << "correctness=" << (result.correctness_ok ? "PASS" : "FAIL") << '\n'
           << "latency_delta_percent=" << result.latency_delta_percent << '\n'
           << "prefill_delta_percent=" << result.prefill_delta_percent << '\n'
           << "decode_delta_percent=" << result.decode_delta_percent << '\n'
           << "throughput_delta_percent=" << result.throughput_delta_percent << '\n'
           << "decision=" << (result.accepted ? "KEEP" : "REJECT") << '\n'
           << "reason=" << result.reason << '\n';
    return output.str();
}

}  // namespace chapter08_profiling
