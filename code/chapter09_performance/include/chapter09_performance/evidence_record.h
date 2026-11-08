// Chapter 09：声明 Benchmark 与 Profiling 共用的测量证据契约。
// 职责：保存实验身份、环境、工作负载、正确性和原始样本，并校验记录完整性。
// 验证：chapter09_evidence_contract_test 覆盖缺失身份、错误结果和非法样本。
#pragma once

#include "chapter09_performance/statistics.h"

#include <cstddef>
#include <string>
#include <vector>

namespace chapter09_performance {

inline constexpr std::size_t kEvidenceSchemaVersion = 2;

struct EnvironmentIdentity {
    std::string model;
    std::string backend;
    std::string device;
    std::string hardware;
    std::size_t hardware_concurrency = 0;
    std::string build_type;
    std::string compiler;
};

struct WorkloadDefinition {
    std::size_t request_count = 0;
    std::size_t prompt_tokens_per_request = 0;
    std::vector<std::size_t> prompt_tokens;
    std::size_t output_tokens_per_request = 0;
    std::size_t warmup_iterations = 0;
    std::size_t measurement_iterations = 0;
    std::string execution_mode;
};

struct MeasurementSample {
    double total_ms = 0.0;
    double prefill_ms = 0.0;
    double decode_ms = 0.0;
    double ttft_ms = 0.0;
    double itl_ms = 0.0;
    double tokens_per_second = 0.0;
    double peak_memory_mb = 0.0;
};

struct EvidenceSummary {
    SampleStatistics total_ms;
    SampleStatistics prefill_ms;
    SampleStatistics decode_ms;
    SampleStatistics ttft_ms;
    SampleStatistics itl_ms;
    SampleStatistics tokens_per_second;
    SampleStatistics peak_memory_mb;
};

struct EvidenceRecord {
    std::size_t schema_version = kEvidenceSchemaVersion;
    std::string experiment_id;
    std::string workload_id;
    EnvironmentIdentity environment;
    WorkloadDefinition workload;
    bool correctness_pass = false;
    std::vector<std::string> correctness_checks;
    std::vector<MeasurementSample> samples;

    bool valid(std::string* error = nullptr) const;
    EvidenceSummary summary() const;
};

std::string render_evidence_json(const EvidenceRecord& record);

}  // namespace chapter09_performance
