// Chapter 07：运行可重复的 CPU 多请求 benchmark，并输出 JSON 结果。
// 职责：复用 Chapter 06 Scheduler/Runtime，比较 1、2、4 个请求的批量行为。
// 验证：每个场景检查请求完成、生成数量、KV Cache 隔离和结果字段完整。
#include "chapter06_generation/embedding.h"
#include "chapter06_generation/position_encoding.h"
#include "chapter06_generation/request.h"
#include "chapter06_generation/runtime.h"
#include "chapter06_generation/sampler.h"
#include "chapter06_generation/scheduler.h"
#include "chapter06_generation/transformer.h"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

enum class ExecutionMode {
    RoundRobin,
    TensorBatch,
};

const char* execution_mode_name(ExecutionMode mode) {
    return mode == ExecutionMode::TensorBatch ? "tensor_batch" : "round_robin";
}

#ifndef CHAPTER07_BUILD_TYPE
#define CHAPTER07_BUILD_TYPE "unknown"
#endif

#ifndef CHAPTER07_COMPILER_ID
#define CHAPTER07_COMPILER_ID "unknown"
#endif

#ifndef CHAPTER07_COMPILER_VERSION
#define CHAPTER07_COMPILER_VERSION "unknown"
#endif

struct MetricStatistics {
    double mean = 0.0;
    double median = 0.0;
    double minimum = 0.0;
    double maximum = 0.0;
};

struct RunMeasurement {
    double prefill_ms = 0.0;
    double decode_ms = 0.0;
    double ttft_ms = 0.0;
    double total_ms = 0.0;
    double throughput_tokens_per_second = 0.0;
    std::size_t generated_tokens = 0;
    std::size_t completed_requests = 0;
    std::size_t model_invocations = 0;
    std::size_t tensor_batch_steps = 0;
    bool correctness = false;
};

struct ScenarioResult {
    ExecutionMode execution_mode = ExecutionMode::RoundRobin;
    std::size_t request_count = 0;
    std::size_t prompt_tokens = 0;
    std::size_t max_new_tokens = 0;
    std::size_t warmup = 0;
    std::size_t iterations = 0;
    double prefill_ms = 0.0;
    double decode_ms = 0.0;
    double total_ms = 0.0;
    double ttft_ms = 0.0;
    double throughput_tokens_per_second = 0.0;
    std::size_t generated_tokens = 0;
    std::size_t completed_requests = 0;
    std::size_t model_invocations = 0;
    std::size_t tensor_batch_steps = 0;
    bool correctness = false;
    MetricStatistics prefill_statistics;
    MetricStatistics decode_statistics;
    MetricStatistics total_statistics;
    MetricStatistics ttft_statistics;
    MetricStatistics throughput_statistics;
    std::vector<RunMeasurement> samples;
};

RunMeasurement run_once(
    std::size_t request_count,
    ExecutionMode execution_mode,
    const chapter06_generation::GenerationConfig& config,
    const chapter06_generation::Transformer& transformer) {
    using namespace chapter06_generation;

    Runtime runtime(config, transformer);
    GreedySampler sampler;
    std::vector<Request> requests;
    std::vector<ScheduledRequest> scheduled;
    requests.reserve(request_count);
    scheduled.reserve(request_count);

    for (std::size_t i = 0; i < request_count; ++i) {
        const TokenId base = static_cast<TokenId>(1 + i);
        requests.emplace_back(TokenSequence{base, base + 1, base + 2, base + 3});
    }
    for (std::size_t i = 0; i < request_count; ++i) {
        scheduled.push_back({i, &requests[i], &sampler});
    }

    const auto prefill_start = Clock::now();
    std::size_t prefetched = 0;
    std::size_t prefill_model_invocations = 0;
    if (execution_mode == ExecutionMode::TensorBatch) {
        std::vector<Request*> batch_requests;
        batch_requests.reserve(requests.size());
        for (Request& request : requests) {
            batch_requests.push_back(&request);
        }
        const BatchPrefillResult result = runtime.prefill_batch(batch_requests);
        prefetched = result.prefilled_requests;
        prefill_model_invocations = result.model_calls;
    } else {
        for (Request& request : requests) {
            prefetched += runtime.prefill(request) > 0 ? 1 : 0;
        }
        prefill_model_invocations = prefetched;
    }
    const auto prefill_end = Clock::now();

    const auto decode_start = Clock::now();
    double first_token_decode_ms = 0.0;
    const auto on_token = [&](std::size_t /*request_id*/, TokenId /*token*/, std::size_t /*step*/) {
            if (first_token_decode_ms == 0.0) {
                first_token_decode_ms =
                    std::chrono::duration<double, std::milli>(Clock::now() - decode_start).count();
            }
        };
    const SchedulerStats stats = execution_mode == ExecutionMode::TensorBatch
        ? TensorBatchScheduler{}.run_with_stats(runtime, scheduled, on_token)
        : RoundRobinScheduler{}.run_with_stats(runtime, scheduled, on_token);
    const auto decode_end = Clock::now();

    bool correct = prefetched == request_count && stats.completed_requests == request_count;
    for (const Request& request : requests) {
        correct = correct && request.finished();
        correct = correct && request.generated_size() == config.max_new_tokens;
        correct = correct && request.cache().size() == request.prompt_size() + request.generated_size();
    }

    RunMeasurement measurement;
    measurement.prefill_ms =
        std::chrono::duration<double, std::milli>(prefill_end - prefill_start).count();
    measurement.decode_ms =
        std::chrono::duration<double, std::milli>(decode_end - decode_start).count();
    measurement.ttft_ms = measurement.prefill_ms + first_token_decode_ms;
    measurement.total_ms = measurement.prefill_ms + measurement.decode_ms;
    measurement.generated_tokens = stats.generated_tokens;
    measurement.completed_requests = stats.completed_requests;
    measurement.model_invocations = prefill_model_invocations +
        (execution_mode == ExecutionMode::TensorBatch
             ? stats.model_batch_calls
             : stats.generated_tokens);
    measurement.tensor_batch_steps = stats.tensor_batch_steps;
    measurement.correctness = correct;
    measurement.throughput_tokens_per_second = measurement.total_ms > 0.0
        ? static_cast<double>(measurement.generated_tokens) * 1000.0 / measurement.total_ms
        : 0.0;
    return measurement;
}

MetricStatistics summarize(std::vector<double> values) {
    MetricStatistics statistics;
    if (values.empty()) {
        return statistics;
    }

    std::sort(values.begin(), values.end());
    for (const double value : values) {
        statistics.mean += value;
    }
    statistics.mean /= static_cast<double>(values.size());
    statistics.minimum = values.front();
    statistics.maximum = values.back();

    const std::size_t middle = values.size() / 2;
    statistics.median = values.size() % 2 == 0
        ? (values[middle - 1] + values[middle]) / 2.0
        : values[middle];
    return statistics;
}

ScenarioResult benchmark_scenario(
    std::size_t request_count,
    ExecutionMode execution_mode,
    std::size_t warmup,
    std::size_t iterations,
    const chapter06_generation::GenerationConfig& config,
    const chapter06_generation::Transformer& transformer) {
    for (std::size_t i = 0; i < warmup; ++i) {
        (void)run_once(request_count, execution_mode, config, transformer);
    }

    ScenarioResult result;
    result.execution_mode = execution_mode;
    result.request_count = request_count;
    result.prompt_tokens = 4;
    result.max_new_tokens = config.max_new_tokens;
    result.warmup = warmup;
    result.iterations = iterations;
    result.correctness = true;
    result.samples.reserve(iterations);

    for (std::size_t i = 0; i < iterations; ++i) {
        const RunMeasurement measurement =
            run_once(request_count, execution_mode, config, transformer);
        result.correctness = result.correctness && measurement.correctness;
        result.samples.push_back(measurement);
    }

    std::vector<double> prefill_samples;
    std::vector<double> decode_samples;
    std::vector<double> total_samples;
    std::vector<double> ttft_samples;
    std::vector<double> throughput_samples;
    prefill_samples.reserve(result.samples.size());
    decode_samples.reserve(result.samples.size());
    total_samples.reserve(result.samples.size());
    ttft_samples.reserve(result.samples.size());
    throughput_samples.reserve(result.samples.size());
    for (const RunMeasurement& sample : result.samples) {
        prefill_samples.push_back(sample.prefill_ms);
        decode_samples.push_back(sample.decode_ms);
        total_samples.push_back(sample.total_ms);
        ttft_samples.push_back(sample.ttft_ms);
        throughput_samples.push_back(sample.throughput_tokens_per_second);
    }

    result.prefill_statistics = summarize(std::move(prefill_samples));
    result.decode_statistics = summarize(std::move(decode_samples));
    result.total_statistics = summarize(std::move(total_samples));
    result.ttft_statistics = summarize(std::move(ttft_samples));
    result.throughput_statistics = summarize(std::move(throughput_samples));
    result.prefill_ms = result.prefill_statistics.mean;
    result.decode_ms = result.decode_statistics.mean;
    result.total_ms = result.total_statistics.mean;
    result.ttft_ms = result.ttft_statistics.mean;
    result.throughput_tokens_per_second = result.throughput_statistics.mean;
    if (!result.samples.empty()) {
        result.generated_tokens = result.samples.front().generated_tokens;
        result.completed_requests = result.samples.front().completed_requests;
        result.model_invocations = result.samples.front().model_invocations;
        result.tensor_batch_steps = result.samples.front().tensor_batch_steps;
    }
    return result;
}

std::string json_number(double value) {
    std::ostringstream output;
    output << std::setprecision(10) << value;
    return output.str();
}

std::string trim(std::string value) {
    const std::size_t first = value.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) {
        return {};
    }
    const std::size_t last = value.find_last_not_of(" \t\r\n");
    return value.substr(first, last - first + 1);
}

std::string cpu_model_name() {
    std::ifstream cpu_info("/proc/cpuinfo");
    std::string line;
    while (std::getline(cpu_info, line)) {
        const std::size_t separator = line.find(':');
        if (separator == std::string::npos) {
            continue;
        }
        const std::string key = trim(line.substr(0, separator));
        if (key == "model name" || key == "Hardware") {
            return trim(line.substr(separator + 1));
        }
    }
    return "unknown";
}

std::string json_escape(const std::string& value) {
    std::ostringstream output;
    for (const char character : value) {
        switch (character) {
        case '\\':
            output << "\\\\";
            break;
        case '"':
            output << "\\\"";
            break;
        case '\n':
            output << "\\n";
            break;
        case '\r':
            output << "\\r";
            break;
        case '\t':
            output << "\\t";
            break;
        default:
            output << character;
            break;
        }
    }
    return output.str();
}

void write_statistics(
    std::ostream& output,
    const std::string& name,
    const MetricStatistics& statistics,
    bool trailing_comma) {
    output << "        \"" << name << "\": {"
           << "\"mean\": " << json_number(statistics.mean)
           << ", \"median\": " << json_number(statistics.median)
           << ", \"min\": " << json_number(statistics.minimum)
           << ", \"max\": " << json_number(statistics.maximum)
           << "}" << (trailing_comma ? "," : "") << "\n";
}

void write_json(
    const std::string& output_path,
    std::size_t warmup,
    std::size_t iterations,
    const std::vector<ScenarioResult>& results) {
    std::ofstream output(output_path);
    if (!output) {
        throw std::runtime_error("cannot open benchmark result: " + output_path);
    }

    output << "{\n"
           << "  \"schema_version\": 2,\n"
           << "  \"experiment_id\": \"EXP-CPU-01\",\n"
           << "  \"workload_id\": \"WL-CPU-TEACHING-BATCH-V1\",\n"
           << "  \"benchmark\": \"chapter07_cpu_multi_request\",\n"
           << "  \"device\": \"cpu\",\n"
           << "  \"environment\": {\n"
           << "    \"cpu_model\": \"" << json_escape(cpu_model_name()) << "\",\n"
           << "    \"hardware_concurrency\": " << std::thread::hardware_concurrency() << ",\n"
           << "    \"compiler\": \"" << json_escape(
                  std::string(CHAPTER07_COMPILER_ID) + " " + CHAPTER07_COMPILER_VERSION) << "\",\n"
           << "    \"build_type\": \"" << json_escape(CHAPTER07_BUILD_TYPE) << "\"\n"
           << "  },\n"
           << "  \"warmup\": " << warmup << ",\n"
           << "  \"iterations\": " << iterations << ",\n"
           << "  \"scenarios\": [\n";
    for (std::size_t i = 0; i < results.size(); ++i) {
        const ScenarioResult& result = results[i];
        output << "    {\n"
               << "      \"execution_mode\": \""
               << execution_mode_name(result.execution_mode) << "\",\n"
               << "      \"request_count\": " << result.request_count << ",\n"
               << "      \"prompt_tokens\": " << result.prompt_tokens << ",\n"
               << "      \"max_new_tokens\": " << result.max_new_tokens << ",\n"
               << "      \"prefill_ms\": " << json_number(result.prefill_ms) << ",\n"
               << "      \"decode_ms\": " << json_number(result.decode_ms) << ",\n"
               << "      \"total_ms\": " << json_number(result.total_ms) << ",\n"
               << "      \"ttft_ms\": " << json_number(result.ttft_ms) << ",\n"
               << "      \"throughput_tokens_per_second\": "
               << json_number(result.throughput_tokens_per_second) << ",\n"
               << "      \"generated_tokens\": " << result.generated_tokens << ",\n"
               << "      \"completed_requests\": " << result.completed_requests << ",\n"
               << "      \"model_invocations\": " << result.model_invocations << ",\n"
               << "      \"tensor_batch_steps\": " << result.tensor_batch_steps << ",\n"
               << "      \"correctness\": " << (result.correctness ? "true" : "false") << ",\n"
               << "      \"statistics\": {\n";
        write_statistics(output, "prefill_ms", result.prefill_statistics, true);
        write_statistics(output, "decode_ms", result.decode_statistics, true);
        write_statistics(output, "total_ms", result.total_statistics, true);
        write_statistics(output, "ttft_ms", result.ttft_statistics, true);
        write_statistics(
            output,
            "throughput_tokens_per_second",
            result.throughput_statistics,
            false);
        output << "      },\n"
               << "      \"samples\": [\n";
        for (std::size_t sample_index = 0; sample_index < result.samples.size(); ++sample_index) {
            const RunMeasurement& sample = result.samples[sample_index];
            output << "        {"
                   << "\"iteration\": " << sample_index + 1
                   << ", \"prefill_ms\": " << json_number(sample.prefill_ms)
                   << ", \"decode_ms\": " << json_number(sample.decode_ms)
                   << ", \"total_ms\": " << json_number(sample.total_ms)
                   << ", \"ttft_ms\": " << json_number(sample.ttft_ms)
                   << ", \"throughput_tokens_per_second\": "
                   << json_number(sample.throughput_tokens_per_second)
                   << ", \"generated_tokens\": " << sample.generated_tokens
                   << ", \"completed_requests\": " << sample.completed_requests
                   << ", \"model_invocations\": " << sample.model_invocations
                   << ", \"tensor_batch_steps\": " << sample.tensor_batch_steps
                   << ", \"correctness\": " << (sample.correctness ? "true" : "false")
                   << "}" << (sample_index + 1 == result.samples.size() ? "\n" : ",\n");
        }
        output << "      ]\n"
               << "    }" << (i + 1 == results.size() ? "\n" : ",\n");
    }
    output << "  ]\n}\n";
}

}  // namespace

int main(int argc, char** argv) {
    try {
        std::string output_path = "chapter07_cpu_multi_request.json";
        std::size_t warmup = 1;
        std::size_t iterations = 3;
        for (int i = 1; i < argc; ++i) {
            const std::string argument = argv[i];
            if (argument == "--output" && i + 1 < argc) {
                output_path = argv[++i];
            } else if (argument == "--warmup" && i + 1 < argc) {
                warmup = static_cast<std::size_t>(std::stoul(argv[++i]));
            } else if (argument == "--iterations" && i + 1 < argc) {
                iterations = static_cast<std::size_t>(std::stoul(argv[++i]));
            } else {
                std::cerr << "usage: multi_request_benchmark [--output PATH] [--warmup N] [--iterations N]\n";
                return 2;
            }
        }
        if (iterations == 0) {
            std::cerr << "iterations must be greater than zero\n";
            return 2;
        }

        chapter06_generation::GenerationConfig config;
        config.max_new_tokens = 4;
        config.max_context_tokens = 32;
        chapter06_generation::Embedding embedding(64);
        chapter06_generation::PositionEncoding position_encoding(32);
        chapter06_generation::Transformer transformer(2, embedding, position_encoding);

        std::vector<ScenarioResult> results;
        for (const ExecutionMode mode : {ExecutionMode::RoundRobin, ExecutionMode::TensorBatch}) {
            for (const std::size_t request_count : {std::size_t{1}, std::size_t{2}, std::size_t{4}}) {
                results.push_back(
                    benchmark_scenario(request_count, mode, warmup, iterations, config, transformer));
            }
        }
        write_json(output_path, warmup, iterations, results);

        bool correct = true;
        for (const ScenarioResult& result : results) {
            correct = correct && result.correctness;
            std::cout << "mode=" << execution_mode_name(result.execution_mode)
                      << " requests=" << result.request_count
                      << " correctness=" << (result.correctness ? "PASS" : "FAIL")
                      << " prefill_ms=" << result.prefill_ms
                      << " decode_ms=" << result.decode_ms
                      << " model_invocations=" << result.model_invocations
                      << " throughput_tokens_per_second=" << result.throughput_tokens_per_second
                      << '\n';
        }
        std::cout << "result_file=" << output_path << '\n';
        return correct ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "benchmark_error=" << error.what() << '\n';
        return 1;
    }
}
