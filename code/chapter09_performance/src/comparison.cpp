// Chapter 09：实现测量记录的可比性检查和保护指标决策。
// 契约：INVALID 与 NOT_COMPARABLE 不计算为性能回退，原始记录保持不变。
// 验证：chapter09_comparison_test 覆盖四类决策和百分比方向。
#include "chapter09_performance/comparison.h"

#include <cmath>
#include <limits>

namespace chapter09_performance {

namespace {

double delta_percent(double baseline, double experiment) {
    if (baseline > 0.0) return (experiment - baseline) * 100.0 / baseline;
    return experiment > 0.0 ? std::numeric_limits<double>::infinity() : 0.0;
}

bool valid_policy(const ComparisonPolicy& policy) {
    const double thresholds[] = {
        policy.max_latency_regression_percent,
        policy.max_throughput_regression_percent,
        policy.max_memory_regression_percent,
    };
    for (const double threshold : thresholds) {
        if (!std::isfinite(threshold) || threshold < 0.0) return false;
    }
    return true;
}

bool same_environment(const EnvironmentIdentity& left, const EnvironmentIdentity& right) {
    return left.model == right.model && left.backend == right.backend &&
           left.device == right.device && left.hardware == right.hardware &&
           left.hardware_concurrency == right.hardware_concurrency &&
           left.build_type == right.build_type && left.compiler == right.compiler;
}

bool same_workload(const WorkloadDefinition& left, const WorkloadDefinition& right) {
    return left.request_count == right.request_count &&
           left.prompt_tokens_per_request == right.prompt_tokens_per_request &&
           left.prompt_tokens == right.prompt_tokens &&
           left.output_tokens_per_request == right.output_tokens_per_request &&
           left.warmup_iterations == right.warmup_iterations &&
           left.measurement_iterations == right.measurement_iterations &&
           left.execution_mode == right.execution_mode;
}

}  // namespace

ComparisonResult compare(const EvidenceRecord& baseline,
                         const EvidenceRecord& experiment,
                         ComparisonPolicy policy) {
    ComparisonResult result;
    std::string error;
    if (!valid_policy(policy)) {
        result.reason = "comparison policy thresholds must be finite and non-negative";
        return result;
    }
    if (!baseline.valid(&error)) {
        result.reason = "invalid baseline: " + error;
        return result;
    }
    if (!experiment.valid(&error)) {
        result.reason = "invalid experiment: " + error;
        return result;
    }
    if (baseline.workload_id != experiment.workload_id ||
        !same_environment(baseline.environment, experiment.environment) ||
        !same_workload(baseline.workload, experiment.workload)) {
        result.decision = Decision::not_comparable;
        result.reason = "workload or environment identity differs";
        return result;
    }

    const EvidenceSummary baseline_summary = baseline.summary();
    const EvidenceSummary experiment_summary = experiment.summary();
    result.latency_delta_percent = delta_percent(
        baseline_summary.total_ms.median, experiment_summary.total_ms.median);
    result.throughput_delta_percent = delta_percent(
        baseline_summary.tokens_per_second.median,
        experiment_summary.tokens_per_second.median);
    result.memory_delta_percent = delta_percent(
        baseline_summary.peak_memory_mb.maximum,
        experiment_summary.peak_memory_mb.maximum);

    const bool latency_ok =
        result.latency_delta_percent <= policy.max_latency_regression_percent;
    const bool throughput_ok =
        result.throughput_delta_percent >= -policy.max_throughput_regression_percent;
    const bool memory_ok =
        result.memory_delta_percent <= policy.max_memory_regression_percent;
    const bool primary_improved =
        policy.primary_metric == PrimaryMetric::latency
            ? result.latency_delta_percent < 0.0
            : result.throughput_delta_percent > 0.0;
    result.decision = primary_improved && latency_ok && throughput_ok && memory_ok
                          ? Decision::keep
                          : Decision::reject;
    if (result.decision == Decision::keep) {
        result.reason = "primary metric improved and protection metrics are within policy";
    } else if (!primary_improved) {
        result.reason = "primary metric did not improve";
    } else {
        result.reason = "a protection metric exceeds policy";
    }
    return result;
}

const char* decision_name(Decision decision) {
    switch (decision) {
        case Decision::keep: return "KEEP";
        case Decision::reject: return "REJECT";
        case Decision::not_comparable: return "NOT_COMPARABLE";
        case Decision::invalid: return "INVALID";
    }
    return "INVALID";
}

}  // namespace chapter09_performance
