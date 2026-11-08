// Chapter 09：实现统一测量记录校验、统计聚合和最小 JSON 输出。
// 契约：失败的正确性闸门和不完整环境不能形成有效性能证据。
// 验证：chapter09_evidence_contract_test 检查字段错误与输出结构。
#include "chapter09_performance/evidence_record.h"

#include <cmath>
#include <iomanip>
#include <sstream>

namespace chapter09_performance {

namespace {

bool missing(const std::string& value) {
    return value.empty() || value == "unknown";
}

bool sample_valid(const MeasurementSample& sample) {
    const double values[] = {
        sample.total_ms,
        sample.prefill_ms,
        sample.decode_ms,
        sample.ttft_ms,
        sample.itl_ms,
        sample.tokens_per_second,
        sample.peak_memory_mb,
    };
    for (double value : values) {
        if (!std::isfinite(value) || value < 0.0) return false;
    }
    return sample.total_ms > 0.0 && sample.tokens_per_second > 0.0;
}

bool prompt_shape_valid(const WorkloadDefinition& workload) {
    if (workload.prompt_tokens.empty()) {
        return workload.prompt_tokens_per_request > 0;
    }
    if (workload.prompt_tokens_per_request != 0) return false;
    if (workload.prompt_tokens.size() != workload.request_count) return false;
    for (const std::size_t token_count : workload.prompt_tokens) {
        if (token_count == 0) return false;
    }
    return true;
}

std::string escape_json(const std::string& value) {
    constexpr char hex[] = "0123456789abcdef";
    std::string escaped;
    escaped.reserve(value.size());
    for (const unsigned char character : value) {
        switch (character) {
            case '"': escaped += "\\\""; break;
            case '\\': escaped += "\\\\"; break;
            case '\b': escaped += "\\b"; break;
            case '\f': escaped += "\\f"; break;
            case '\n': escaped += "\\n"; break;
            case '\r': escaped += "\\r"; break;
            case '\t': escaped += "\\t"; break;
            default:
                if (character < 0x20) {
                    escaped += "\\u00";
                    escaped.push_back(hex[character >> 4]);
                    escaped.push_back(hex[character & 0x0f]);
                } else {
                    escaped.push_back(static_cast<char>(character));
                }
                break;
        }
    }
    return escaped;
}

std::vector<double> collect(const std::vector<MeasurementSample>& samples,
                            double MeasurementSample::*field) {
    std::vector<double> values;
    values.reserve(samples.size());
    for (const auto& sample : samples) values.push_back(sample.*field);
    return values;
}

}  // namespace

bool EvidenceRecord::valid(std::string* error) const {
    const auto fail = [error](const char* message) {
        if (error != nullptr) *error = message;
        return false;
    };
    if (schema_version != kEvidenceSchemaVersion) {
        return fail("unsupported schema_version");
    }
    if (missing(experiment_id)) return fail("missing experiment_id");
    if (missing(workload_id)) return fail("missing workload_id");
    if (missing(environment.model) || missing(environment.backend) ||
        missing(environment.device) || missing(environment.hardware) ||
        environment.hardware_concurrency == 0 || missing(environment.build_type) ||
        missing(environment.compiler)) {
        return fail("incomplete environment identity");
    }
    if (workload.request_count == 0 || !prompt_shape_valid(workload) ||
        workload.output_tokens_per_request == 0 ||
        workload.measurement_iterations == 0 || missing(workload.execution_mode)) {
        return fail("incomplete workload definition");
    }
    if (!correctness_pass) return fail("correctness gate failed");
    if (correctness_checks.empty()) return fail("missing correctness checks");
    for (const auto& check : correctness_checks) {
        if (missing(check)) return fail("empty correctness check");
    }
    if (samples.size() != workload.measurement_iterations) {
        return fail("sample count does not match measurement_iterations");
    }
    for (const auto& sample : samples) {
        if (!sample_valid(sample)) return fail("invalid measurement sample");
    }
    return true;
}

EvidenceSummary EvidenceRecord::summary() const {
    EvidenceSummary result;
    result.total_ms = summarize(collect(samples, &MeasurementSample::total_ms));
    result.prefill_ms = summarize(collect(samples, &MeasurementSample::prefill_ms));
    result.decode_ms = summarize(collect(samples, &MeasurementSample::decode_ms));
    result.ttft_ms = summarize(collect(samples, &MeasurementSample::ttft_ms));
    result.itl_ms = summarize(collect(samples, &MeasurementSample::itl_ms));
    result.tokens_per_second = summarize(
        collect(samples, &MeasurementSample::tokens_per_second));
    result.peak_memory_mb = summarize(collect(samples, &MeasurementSample::peak_memory_mb));
    return result;
}

std::string render_evidence_json(const EvidenceRecord& record) {
    const EvidenceSummary aggregate = record.summary();
    std::ostringstream output;
    output << std::fixed << std::setprecision(4)
           << "{\n"
           << "  \"schema_version\": " << record.schema_version << ",\n"
           << "  \"experiment_id\": \"" << escape_json(record.experiment_id) << "\",\n"
           << "  \"workload_id\": \"" << escape_json(record.workload_id) << "\",\n"
           << "  \"environment\": {\"model\": \""
           << escape_json(record.environment.model) << "\", \"backend\": \""
           << escape_json(record.environment.backend) << "\", \"device\": \""
           << escape_json(record.environment.device) << "\", \"hardware\": \""
           << escape_json(record.environment.hardware)
           << "\", \"hardware_concurrency\": "
           << record.environment.hardware_concurrency << ", \"build_type\": \""
           << escape_json(record.environment.build_type) << "\", \"compiler\": \""
           << escape_json(record.environment.compiler) << "\"},\n"
           << "  \"workload\": {\"request_count\": " << record.workload.request_count
           << ", \"prompt_tokens_per_request\": "
           << record.workload.prompt_tokens_per_request
           << ", \"prompt_tokens\": [";
    for (std::size_t index = 0; index < record.workload.prompt_tokens.size(); ++index) {
        output << record.workload.prompt_tokens[index]
               << (index + 1 == record.workload.prompt_tokens.size() ? "" : ", ");
    }
    output << "]"
           << ", \"output_tokens_per_request\": "
           << record.workload.output_tokens_per_request
           << ", \"warmup_iterations\": " << record.workload.warmup_iterations
           << ", \"measurement_iterations\": "
           << record.workload.measurement_iterations << ", \"execution_mode\": \""
           << escape_json(record.workload.execution_mode) << "\"},\n"
           << "  \"correctness_pass\": "
           << (record.correctness_pass ? "true" : "false") << ",\n"
           << "  \"correctness_checks\": [";
    for (std::size_t index = 0; index < record.correctness_checks.size(); ++index) {
        output << "\"" << escape_json(record.correctness_checks[index]) << "\""
               << (index + 1 == record.correctness_checks.size() ? "" : ", ");
    }
    output << "],\n"
           << "  \"statistics\": {\"total_ms_median\": "
           << aggregate.total_ms.median << ", \"tokens_per_second_median\": "
           << aggregate.tokens_per_second.median << ", \"peak_memory_mb_max\": "
           << aggregate.peak_memory_mb.maximum << "},\n"
           << "  \"samples\": [\n";
    for (std::size_t index = 0; index < record.samples.size(); ++index) {
        const auto& sample = record.samples[index];
        output << "    {\"total_ms\": " << sample.total_ms
               << ", \"prefill_ms\": " << sample.prefill_ms
               << ", \"decode_ms\": " << sample.decode_ms
               << ", \"ttft_ms\": " << sample.ttft_ms
               << ", \"itl_ms\": " << sample.itl_ms
               << ", \"tokens_per_second\": " << sample.tokens_per_second
               << ", \"peak_memory_mb\": " << sample.peak_memory_mb << "}"
               << (index + 1 == record.samples.size() ? "\n" : ",\n");
    }
    output << "  ]\n}\n";
    return output.str();
}

}  // namespace chapter09_performance
