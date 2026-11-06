// Chapter 09：声明正确性优先的性能对比策略和决策记录。
// 职责：验证两份测量是否可比，再根据延迟、吞吐和内存保护阈值给出结论。
// 验证：chapter09_comparison_test 覆盖 KEEP、REJECT、NOT_COMPARABLE 和 INVALID。
#pragma once

#include "chapter09_performance/evidence_record.h"

#include <string>

namespace chapter09_performance {

enum class Decision {
    keep,
    reject,
    not_comparable,
    invalid,
};

enum class PrimaryMetric {
    latency,
    throughput,
};

struct ComparisonPolicy {
    PrimaryMetric primary_metric = PrimaryMetric::throughput;
    double max_latency_regression_percent = 5.0;
    double max_throughput_regression_percent = 5.0;
    double max_memory_regression_percent = 5.0;
};

struct ComparisonResult {
    Decision decision = Decision::invalid;
    double latency_delta_percent = 0.0;
    double throughput_delta_percent = 0.0;
    double memory_delta_percent = 0.0;
    std::string reason;
};

ComparisonResult compare(const EvidenceRecord& baseline,
                         const EvidenceRecord& experiment,
                         ComparisonPolicy policy = {});
const char* decision_name(Decision decision);

}  // namespace chapter09_performance
