// Chapter 08：声明 correctness-first 的 profiling A/B 判定。
// 契约：先检查 workload、backend 和 token budget 可比，再比较阶段与总延迟。
// 验证：Chapter08 CTest 覆盖正确性失败、条件不匹配和延迟回退。
#pragma once

#include "chapter08_profiling/profile_record.h"

#include <string>

namespace chapter08_profiling {

struct ComparisonPolicy {
    double max_latency_regression_percent = 5.0;
};

struct ComparisonResult {
    bool accepted = false;
    bool correctness_ok = false;
    double latency_delta_percent = 0.0;
    double prefill_delta_percent = 0.0;
    double decode_delta_percent = 0.0;
    double throughput_delta_percent = 0.0;
    std::string reason;
};

ComparisonResult compare(const ProfileRecord& baseline,
                         const ProfileRecord& experiment,
                         ComparisonPolicy policy = {});
std::string render_comparison(const ComparisonResult& result);

}  // namespace chapter08_profiling
