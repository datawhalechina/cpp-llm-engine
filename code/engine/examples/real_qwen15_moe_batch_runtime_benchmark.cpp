// Chapter 09：比较不同 batch 下 Qwen1.5-MoE 逐行与分组 Decode。
// 计时边界：不含模型加载和 Prefill，包含 24 层 Decoder 与 LM Head。
#include "cpp_llm_engine/real_qwen2_moe_executor_factory.h"

#include "chapter06_generation/sampler.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

struct Result {
    double milliseconds = 0.0;
    std::vector<std::vector<chapter06_generation::Logits>> logits_by_step;
    std::vector<chapter06_generation::TokenSequence> generated;
};

bool run(cpp_llm_engine::RealQwen2MoeExecutorFactory& factory,
         cpp_llm_engine::RealQwen2MoeBatchMode mode,
         std::size_t request_seed,
         std::size_t batch,
         std::size_t decode_steps,
         Result* result,
         std::string* error) {
    std::vector<std::shared_ptr<chapter06_generation::RequestModelExecutor>>
        executors;
    executors.reserve(batch);
    for (std::size_t index = 0; index < batch; ++index) {
        executors.push_back(factory.create_executor());
    }
    chapter06_generation::GreedySampler sampler;
    chapter06_generation::TokenSequence next_tokens;
    for (const auto& executor : executors) {
        chapter06_generation::Logits logits;
        if (executor == nullptr || !executor->prefill({0}, &logits, error)) {
            return false;
        }
        next_tokens.push_back(sampler.sample(logits));
    }
    result->generated.resize(batch);
    std::vector<std::size_t> expected_positions(batch, 1);
    std::vector<std::size_t> request_ids(batch);
    for (std::size_t index = 0; index < batch; ++index) {
        request_ids[index] = request_seed + index;
    }
    factory.set_moe_batch_mode(mode);
    const auto start = Clock::now();
    for (std::size_t step = 0; step < decode_steps; ++step) {
        std::vector<chapter06_generation::Logits> logits;
        if (!factory.decode_batch(
                executors, next_tokens, expected_positions, request_ids,
                &logits, error)) {
            return false;
        }
        result->logits_by_step.push_back(logits);
        for (std::size_t index = 0; index < batch; ++index) {
            next_tokens[index] = sampler.sample(logits[index]);
            result->generated[index].push_back(next_tokens[index]);
            ++expected_positions[index];
        }
    }
    const auto stop = Clock::now();
    result->milliseconds =
        std::chrono::duration<double, std::milli>(stop - start).count();
    return true;
}

double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    const std::size_t middle = values.size() / 2;
    return values.size() % 2 == 0
        ? (values[middle - 1] + values[middle]) * 0.5
        : values[middle];
}

float maximum_error(
    const std::vector<std::vector<chapter06_generation::Logits>>& left,
    const std::vector<std::vector<chapter06_generation::Logits>>& right) {
    if (left.size() != right.size()) return INFINITY;
    float maximum = 0.0F;
    for (std::size_t step = 0; step < left.size(); ++step) {
        if (left[step].size() != right[step].size()) return INFINITY;
        for (std::size_t row = 0; row < left[step].size(); ++row) {
            if (left[step][row].size() != right[step][row].size()) {
                return INFINITY;
            }
            for (std::size_t column = 0;
                 column < left[step][row].size(); ++column) {
                maximum = std::max(
                    maximum, std::fabs(left[step][row][column] -
                                       right[step][row][column]));
            }
        }
    }
    return maximum;
}

void print_samples(const char* name, const std::vector<double>& samples) {
    std::cout << name << "_samples_ms:";
    for (const double sample : samples) std::cout << ' ' << sample;
    std::cout << '\n';
}

std::vector<std::size_t> parse_sizes(const std::string& text) {
    std::vector<std::size_t> values;
    std::size_t begin = 0;
    while (begin <= text.size()) {
        const std::size_t end = text.find(',', begin);
        const std::string item = text.substr(
            begin, end == std::string::npos ? end : end - begin);
        if (item.empty()) return {};
        const std::size_t value = static_cast<std::size_t>(std::stoul(item));
        if (value == 0) return {};
        values.push_back(value);
        if (end == std::string::npos) break;
        begin = end + 1;
    }
    return values;
}

bool measure_cell(cpp_llm_engine::RealQwen2MoeExecutorFactory& factory,
                  std::size_t batch,
                  std::size_t decode_steps,
                  std::size_t iterations,
                  std::size_t* request_seed,
                  std::string* error) {
    std::vector<double> row_wise_samples;
    std::vector<double> grouped_samples;
    float max_error = 0.0F;
    bool tokens_match = true;
    for (std::size_t iteration = 0; iteration < iterations; ++iteration) {
        Result first;
        Result second;
        const bool grouped_first = iteration % 2 == 1;
        const auto first_mode = grouped_first
            ? cpp_llm_engine::RealQwen2MoeBatchMode::Grouped
            : cpp_llm_engine::RealQwen2MoeBatchMode::RowWise;
        const auto second_mode = grouped_first
            ? cpp_llm_engine::RealQwen2MoeBatchMode::RowWise
            : cpp_llm_engine::RealQwen2MoeBatchMode::Grouped;
        if (!run(factory, first_mode, *request_seed, batch, decode_steps,
                 &first, error)) {
            return false;
        }
        *request_seed += batch;
        if (!run(factory, second_mode, *request_seed, batch, decode_steps,
                 &second, error)) {
            return false;
        }
        *request_seed += batch;
        const auto& row_wise = grouped_first ? second : first;
        const auto& grouped = grouped_first ? first : second;
        row_wise_samples.push_back(row_wise.milliseconds);
        grouped_samples.push_back(grouped.milliseconds);
        max_error = std::max(
            max_error, maximum_error(row_wise.logits_by_step,
                                     grouped.logits_by_step));
        tokens_match = tokens_match &&
            row_wise.generated == grouped.generated;
        std::cout << "matrix_sample batch=" << batch
                  << " decode_steps=" << decode_steps
                  << " iteration=" << iteration + 1
                  << " path_order="
                  << (grouped_first ? "grouped-first" : "row-wise-first")
                  << " row_wise_ms=" << row_wise.milliseconds
                  << " grouped_ms=" << grouped.milliseconds << '\n';
    }

    const double row_wise_median = median(row_wise_samples);
    const double grouped_median = median(grouped_samples);
    const bool correct = tokens_match && max_error <= 1.0e-4F;
    std::cout << "matrix_cell batch=" << batch
              << " decode_steps=" << decode_steps
              << " row_wise_median_ms=" << row_wise_median
              << " grouped_median_ms=" << grouped_median
              << " grouped_to_row_wise_ratio="
              << grouped_median / row_wise_median
              << " maximum_logits_error=" << max_error
              << " tokens_match=" << (tokens_match ? "true" : "false")
              << " correctness=" << (correct ? "PASS" : "FAIL") << '\n';
    print_samples("row_wise", row_wise_samples);
    print_samples("grouped", grouped_samples);
    return correct;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2 || argc > 5) {
        std::cerr << "usage: real_qwen15_moe_batch_runtime_benchmark "
                     "<model.gguf> [iterations=3] [batches=2] "
                     "[decode_steps=1]\n";
        return 2;
    }
    const std::size_t iterations = argc >= 3
        ? static_cast<std::size_t>(std::stoul(argv[2])) : 3;
    const auto batches = parse_sizes(argc >= 4 ? argv[3] : "2");
    const auto decode_steps = parse_sizes(argc >= 5 ? argv[4] : "1");
    if (iterations == 0 || batches.empty() || decode_steps.empty()) {
        std::cerr << "iterations, batches, and decode steps must be positive\n";
        return 2;
    }
    std::string error;
    auto factory = cpp_llm_engine::RealQwen2MoeExecutorFactory::load(
        argv[1], &error);
    if (factory == nullptr) {
        std::cerr << "Qwen1.5-MoE load failed: " << error << '\n';
        return 1;
    }
    factory->set_layer_cache_capacity(factory->layer_count());

    std::cout << "iterations: " << iterations << '\n'
              << "cache_layers: " << factory->layer_count() << '\n';
    std::size_t request_seed = 1;
    for (const std::size_t batch : batches) {
        for (const std::size_t steps : decode_steps) {
            if (!measure_cell(*factory, batch, steps, iterations,
                              &request_seed, &error)) {
                std::cerr << "Qwen1.5-MoE runtime matrix failed: "
                          << error << '\n';
                return 1;
            }
        }
    }
    std::cout << "matrix_correctness: PASS\n";
    return 0;
}
