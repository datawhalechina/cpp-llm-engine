// Chapter 09：提供性能契约测试共用的完整测量记录。
// 职责：减少测试样板，调用方只修改当前场景关心的字段。
// 验证：由 Chapter09 evidence 与 comparison 测试共同覆盖。
#pragma once

#include "chapter09_performance/evidence_record.h"

namespace chapter09_performance::testing {

inline EvidenceRecord sample_record(const std::string& experiment_id) {
    EvidenceRecord record;
    record.experiment_id = experiment_id;
    record.workload_id = "WL-CPU-BATCH-V1";
    record.environment = {
        "TinyLlama-1.1B-Chat-Q4_K_M", "cpu", "x86_64-avx2",
        "AMD Ryzen 9 9955HX", 32, "Release", "gcc-13"};
    record.workload.request_count = 2;
    record.workload.prompt_tokens_per_request = 8;
    record.workload.output_tokens_per_request = 4;
    record.workload.warmup_iterations = 1;
    record.workload.measurement_iterations = 3;
    record.workload.execution_mode = "tensor_batch";
    record.correctness_pass = true;
    record.correctness_checks = {"golden_tokens", "batch_matches_sequential"};
    record.samples = {
        {10.0, 4.0, 5.0, 4.5, 1.8, 800.0, 512.0},
        {11.0, 4.2, 5.5, 4.7, 1.9, 760.0, 514.0},
        {9.0, 3.8, 4.5, 4.3, 1.7, 840.0, 510.0},
    };
    return record;
}

}  // namespace chapter09_performance::testing
