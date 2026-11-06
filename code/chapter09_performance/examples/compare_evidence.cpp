// Chapter 09：读取完整规范化证据并交给统一 C++ 校验与对比策略。
// 输入：由 compare_evidence_pair.mjs 生成的两份临时 wire record。
// 验证：chapter09_real_evidence_bridge_test 覆盖 KEEP、不可比和非法证据。
#include "chapter09_performance/comparison.h"

#include <cctype>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

std::string percent_decode(const std::string& value) {
    std::string decoded;
    decoded.reserve(value.size());
    for (std::size_t index = 0; index < value.size(); ++index) {
        if (value[index] != '%' || index + 2 >= value.size()) {
            decoded.push_back(value[index]);
            continue;
        }
        const auto hex = [](char digit) -> int {
            if (digit >= '0' && digit <= '9') return digit - '0';
            digit = static_cast<char>(std::tolower(static_cast<unsigned char>(digit)));
            if (digit >= 'a' && digit <= 'f') return digit - 'a' + 10;
            return -1;
        };
        const int high = hex(value[index + 1]);
        const int low = hex(value[index + 2]);
        if (high < 0 || low < 0) throw std::runtime_error("invalid percent encoding");
        decoded.push_back(static_cast<char>((high << 4) | low));
        index += 2;
    }
    return decoded;
}

std::vector<std::string> split(const std::string& value, char delimiter) {
    std::vector<std::string> fields;
    std::istringstream input(value);
    std::string field;
    while (std::getline(input, field, delimiter)) fields.push_back(field);
    return fields;
}

std::size_t size_value(const std::unordered_map<std::string, std::string>& fields,
                       const char* key) {
    const auto iterator = fields.find(key);
    return iterator == fields.end() ? 0 : static_cast<std::size_t>(std::stoull(iterator->second));
}

double double_value(const std::string& value) {
    std::size_t consumed = 0;
    const double result = std::stod(value, &consumed);
    if (consumed != value.size()) throw std::runtime_error("invalid sample number");
    return result;
}

chapter09_performance::EvidenceRecord read_record(const std::string& path) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("cannot open wire record: " + path);

    std::unordered_map<std::string, std::string> fields;
    std::vector<chapter09_performance::MeasurementSample> samples;
    std::string line;
    while (std::getline(input, line)) {
        const std::size_t separator = line.find('\t');
        if (separator == std::string::npos) throw std::runtime_error("invalid wire record line");
        const std::string key = line.substr(0, separator);
        const std::string value = line.substr(separator + 1);
        if (key != "sample") {
            fields[key] = value;
            continue;
        }
        const auto values = split(value, ',');
        if (values.size() != 7) throw std::runtime_error("sample must contain seven values");
        samples.push_back({double_value(values[0]), double_value(values[1]),
                           double_value(values[2]), double_value(values[3]),
                           double_value(values[4]), double_value(values[5]),
                           double_value(values[6])});
    }

    const auto text = [&fields](const char* key) {
        const auto iterator = fields.find(key);
        return iterator == fields.end() ? std::string{} : percent_decode(iterator->second);
    };
    chapter09_performance::EvidenceRecord record;
    record.schema_version = size_value(fields, "schema_version");
    record.experiment_id = text("experiment_id");
    record.workload_id = text("workload_id");
    record.environment = {text("model"), text("backend"), text("device"),
                          text("build_type"), text("compiler")};
    record.workload.request_count = size_value(fields, "request_count");
    record.workload.prompt_tokens_per_request = size_value(fields, "prompt_tokens_per_request");
    for (const auto& token : split(fields["prompt_tokens"], ',')) {
        if (!token.empty()) {
            record.workload.prompt_tokens.push_back(static_cast<std::size_t>(std::stoull(token)));
        }
    }
    record.workload.output_tokens_per_request = size_value(fields, "output_tokens_per_request");
    record.workload.warmup_iterations = size_value(fields, "warmup_iterations");
    record.workload.measurement_iterations = size_value(fields, "measurement_iterations");
    record.workload.execution_mode = text("execution_mode");
    record.correctness_pass = fields["correctness_pass"] == "1";
    record.correctness_checks.assign(size_value(fields, "correctness_check_count"), "wire-check");
    record.samples = std::move(samples);
    return record;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "usage: chapter09_compare_evidence BASELINE.wire EXPERIMENT.wire\n";
        return 2;
    }
    try {
        const auto result = chapter09_performance::compare(
            read_record(argv[1]), read_record(argv[2]));
        std::cout << std::fixed << std::setprecision(4)
                  << "{\"decision\":\"" << chapter09_performance::decision_name(result.decision)
                  << "\",\"latency_delta_percent\":" << result.latency_delta_percent
                  << ",\"throughput_delta_percent\":" << result.throughput_delta_percent
                  << ",\"memory_delta_percent\":" << result.memory_delta_percent
                  << ",\"reason\":\"" << result.reason << "\"}\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "comparison input failed: " << error.what() << '\n';
        return 2;
    }
}
