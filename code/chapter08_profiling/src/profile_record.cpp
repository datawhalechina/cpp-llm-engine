// Chapter 08：把 benchmark key=value 输出解析为统一 profiling 记录。
// 契约：未知字段忽略，教学基线与真实 ServiceCore 基线使用同一报告入口。
// 验证：单元测试覆盖派生占比、瓶颈范围和非法测量值。
#include "chapter08_profiling/profile_record.h"

#include <cstdlib>
#include <iomanip>
#include <istream>
#include <sstream>

namespace chapter08_profiling {

namespace {

void assign_value(ProfileRecord& record, const std::string& key, const std::string& value) {
    if (key == "experiment_id") {
        record.experiment_id = value;
    } else if (key == "workload_id" || key == "runtime_benchmark") {
        record.workload_id = value;
    } else if (key == "backend") {
        record.backend = value;
    } else if (key == "model") {
        record.model = value;
    } else if (key == "cpu_model") {
        record.cpu_model = value;
    } else if (key == "build_type") {
        record.build_type = value;
    } else if (key == "execution_pattern") {
        record.execution_pattern = value;
    } else if (key == "correctness") {
        record.correctness_pass = value == "PASS";
    } else if (key == "iterations") {
        record.iterations = static_cast<std::size_t>(std::strtoull(value.c_str(), nullptr, 10));
    } else if (key == "request_count") {
        record.request_count =
            static_cast<std::size_t>(std::strtoull(value.c_str(), nullptr, 10));
    } else if (key == "generated_tokens_per_iteration") {
        record.generated_tokens_per_iteration =
            static_cast<std::size_t>(std::strtoull(value.c_str(), nullptr, 10));
    } else if (key == "prefill_ms") {
        record.prefill_ms = std::strtod(value.c_str(), nullptr);
    } else if (key == "decode_ms") {
        record.decode_ms = std::strtod(value.c_str(), nullptr);
    } else if (key == "runtime_overhead_ms") {
        record.runtime_overhead_ms = std::strtod(value.c_str(), nullptr);
    } else if (key == "total_ms") {
        record.total_ms = std::strtod(value.c_str(), nullptr);
    } else if (key == "ttft_ms") {
        record.ttft_ms = std::strtod(value.c_str(), nullptr);
    } else if (key == "completion_ms") {
        record.completion_ms = std::strtod(value.c_str(), nullptr);
    } else if (key == "ttft_spread_ms") {
        record.ttft_spread_ms = std::strtod(value.c_str(), nullptr);
    } else if (key == "completion_spread_ms") {
        record.completion_spread_ms = std::strtod(value.c_str(), nullptr);
    } else if (key == "reported_tokens_per_second") {
        record.reported_tokens_per_second = std::strtod(value.c_str(), nullptr);
    } else if (key == "model_weight_payload_bytes") {
        record.model_weight_payload_bytes =
            static_cast<std::size_t>(std::strtoull(value.c_str(), nullptr, 10));
    } else if (key == "kv_cache_bytes") {
        record.kv_cache_bytes =
            static_cast<std::size_t>(std::strtoull(value.c_str(), nullptr, 10));
    } else if (key == "peak_memory_mb") {
        record.peak_memory_mb = std::strtod(value.c_str(), nullptr);
    }
}

}  // namespace

double ProfileRecord::tokens_per_second() const {
    if (reported_tokens_per_second > 0.0) return reported_tokens_per_second;
    if (total_ms <= 0.0 || generated_tokens_per_iteration == 0) {
        return 0.0;
    }
    return static_cast<double>(generated_tokens_per_iteration) * 1000.0 / total_ms;
}

double ProfileRecord::model_compute_share_percent() const {
    if (total_ms <= 0.0) return 0.0;
    return (prefill_ms + decode_ms) * 100.0 / total_ms;
}

double ProfileRecord::runtime_overhead_share_percent() const {
    if (total_ms <= 0.0) return 0.0;
    return runtime_overhead_ms * 100.0 / total_ms;
}

bool ProfileRecord::valid(std::string* error) const {
    const auto fail = [error](const char* message) {
        if (error != nullptr) {
            *error = message;
        }
        return false;
    };
    if (workload_id.empty() || workload_id == "unknown") return fail("missing workload_id");
    if (backend.empty() || backend == "unknown") return fail("missing backend");
    if (!correctness_pass) return fail("correctness failed");
    if (iterations == 0) return fail("iterations must be positive");
    if (total_ms <= 0.0) return fail("total_ms must be positive");
    if (prefill_ms < 0.0 || decode_ms < 0.0 || runtime_overhead_ms < 0.0 ||
        ttft_ms < 0.0 || completion_ms < 0.0 || ttft_spread_ms < 0.0 ||
        completion_spread_ms < 0.0 || reported_tokens_per_second < 0.0 ||
        peak_memory_mb < 0.0) {
        return fail("measurements must be non-negative");
    }
    return true;
}

ProfileRecord parse_benchmark_output(std::istream& input) {
    ProfileRecord record;
    std::string line;
    while (std::getline(input, line)) {
        const std::size_t separator = line.find('=');
        if (separator == std::string::npos) continue;
        const std::string key = line.substr(0, separator);
        const std::string value = line.substr(separator + 1);
        assign_value(record, key, value);
    }
    if (record.backend == "unknown" && record.workload_id.find("cpu") != std::string::npos) {
        record.backend = "cpu";
    }
    if (record.total_ms <= 0.0) {
        record.total_ms = record.prefill_ms + record.decode_ms;
    }
    return record;
}

std::string render_report(const ProfileRecord& record) {
    std::ostringstream output;
    output << std::fixed << std::setprecision(4);
    if (record.experiment_id != "unregistered") {
        output << "experiment_id=" << record.experiment_id << '\n';
    }
    output << "workload_id=" << record.workload_id << '\n'
           << "backend=" << record.backend << '\n'
           << "correctness=" << (record.correctness_pass ? "PASS" : "FAIL") << '\n'
           << "iterations=" << record.iterations << '\n'
           << "request_count=" << record.request_count << '\n'
           << "generated_tokens_per_iteration="
           << record.generated_tokens_per_iteration << '\n';
    if (record.model != "unknown") output << "model=" << record.model << '\n';
    if (record.cpu_model != "unknown") output << "cpu_model=" << record.cpu_model << '\n';
    if (record.build_type != "unknown") output << "build_type=" << record.build_type << '\n';
    output
           << "prefill_ms=" << record.prefill_ms << '\n'
           << "decode_ms=" << record.decode_ms << '\n'
           << "runtime_overhead_ms=" << record.runtime_overhead_ms << '\n'
           << "total_ms=" << record.total_ms << '\n'
           << "ttft_ms=" << record.ttft_ms << '\n'
           << "completion_ms=" << record.completion_ms << '\n'
           << "ttft_spread_ms=" << record.ttft_spread_ms << '\n'
           << "completion_spread_ms=" << record.completion_spread_ms << '\n'
           << "tokens_per_second=" << record.tokens_per_second() << '\n';
    const double measured_phase_ms = record.prefill_ms + record.decode_ms;
    if (measured_phase_ms > 0.0) {
        const double prefill_share = record.prefill_ms * 100.0 / measured_phase_ms;
        const double decode_share = record.decode_ms * 100.0 / measured_phase_ms;
        output << "prefill_share_percent=" << prefill_share << '\n'
               << "decode_share_percent=" << decode_share << '\n'
               << "model_compute_share_percent="
               << record.model_compute_share_percent() << '\n'
               << "runtime_overhead_share_percent="
               << record.runtime_overhead_share_percent() << '\n'
               << "dominant_phase=" << (record.prefill_ms >= record.decode_ms ? "prefill" : "decode")
               << '\n'
               << "bottleneck_scope="
               << (record.model_compute_share_percent() >= 90.0
                       ? "model_compute" : "runtime_or_service")
               << '\n';
    }
    if (record.execution_pattern != "unknown") {
        output << "execution_pattern=" << record.execution_pattern << '\n';
    }
    if (record.peak_memory_mb > 0.0) {
        output << "peak_memory_mb=" << record.peak_memory_mb << '\n';
    }
    if (record.model_weight_payload_bytes > 0) {
        output << "model_weight_payload_bytes="
               << record.model_weight_payload_bytes << '\n';
    }
    if (record.kv_cache_bytes > 0) {
        output << "kv_cache_bytes=" << record.kv_cache_bytes << '\n';
    }
    return output.str();
}

}  // namespace chapter08_profiling
