// Chapter 09：比较 DeepSeek-V2-Lite 双请求逐行与分组 MoE Decode。
// 计时边界：不含模型加载和 Prefill，包含 27 层 Decoder 与 LM Head。
#include "cpp_llm_engine/real_deepseek2_executor_factory.h"

#include "chapter06_generation/sampler.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

struct Result {
    double milliseconds = 0.0;
    std::vector<chapter06_generation::Logits> logits;
    chapter06_generation::TokenSequence tokens;
};

bool run(cpp_llm_engine::RealDeepSeek2ExecutorFactory& factory,
         cpp_llm_engine::RealDeepSeek2MoeBatchMode mode,
         std::size_t request_seed,
         Result* result,
         std::string* error) {
    constexpr std::size_t kBatch = 2;
    std::vector<std::shared_ptr<chapter06_generation::RequestModelExecutor>>
        executors;
    chapter06_generation::TokenSequence decode_tokens;
    chapter06_generation::GreedySampler sampler;
    for (std::size_t request = 0; request < kBatch; ++request) {
        auto executor = factory.create_executor();
        chapter06_generation::Logits logits;
        if (executor == nullptr || !executor->prefill({0}, &logits, error)) {
            return false;
        }
        decode_tokens.push_back(sampler.sample(logits));
        executors.push_back(std::move(executor));
    }

    factory.set_moe_batch_mode(mode);
    const std::vector<std::size_t> positions(kBatch, 1);
    const std::vector<std::size_t> request_ids{
        request_seed, request_seed + 1};
    const auto start = Clock::now();
    std::vector<chapter06_generation::Logits> logits;
    if (!factory.decode_batch(executors, decode_tokens, positions, request_ids,
                              &logits, error)) {
        return false;
    }
    const auto stop = Clock::now();

    result->milliseconds =
        std::chrono::duration<double, std::milli>(stop - start).count();
    result->logits = std::move(logits);
    for (const auto& row : result->logits) {
        result->tokens.push_back(sampler.sample(row));
    }
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
    const std::vector<chapter06_generation::Logits>& left,
    const std::vector<chapter06_generation::Logits>& right) {
    if (left.size() != right.size()) return INFINITY;
    float maximum = 0.0F;
    for (std::size_t row = 0; row < left.size(); ++row) {
        if (left[row].size() != right[row].size()) return INFINITY;
        for (std::size_t column = 0; column < left[row].size(); ++column) {
            maximum = std::max(
                maximum, std::fabs(left[row][column] - right[row][column]));
        }
    }
    return maximum;
}

double minimum_cosine_similarity(
    const std::vector<chapter06_generation::Logits>& left,
    const std::vector<chapter06_generation::Logits>& right) {
    if (left.size() != right.size()) return -1.0;
    double minimum = 1.0;
    for (std::size_t row = 0; row < left.size(); ++row) {
        if (left[row].size() != right[row].size()) return -1.0;
        double dot = 0.0;
        double left_norm = 0.0;
        double right_norm = 0.0;
        for (std::size_t column = 0; column < left[row].size(); ++column) {
            dot += static_cast<double>(left[row][column]) * right[row][column];
            left_norm += static_cast<double>(left[row][column]) *
                left[row][column];
            right_norm += static_cast<double>(right[row][column]) *
                right[row][column];
        }
        if (left_norm == 0.0 || right_norm == 0.0) return -1.0;
        minimum = std::min(
            minimum, dot / std::sqrt(left_norm * right_norm));
    }
    return minimum;
}

void print_samples(const char* name, const std::vector<double>& samples) {
    std::cout << name << "_samples_ms:";
    for (const double sample : samples) std::cout << ' ' << sample;
    std::cout << '\n';
}

}  // namespace

int main(int argc, char** argv) {
    std::cout << std::setprecision(9);
    if (argc < 2 || argc > 3) {
        std::cerr << "usage: real_deepseek2_moe_batch_runtime_benchmark "
                     "<model.gguf> [iterations=3]\n";
        return 2;
    }
    const std::size_t iterations = argc == 3
        ? static_cast<std::size_t>(std::stoul(argv[2])) : 3;
    if (iterations == 0) {
        std::cerr << "iterations must be positive\n";
        return 2;
    }

    std::string error;
    auto factory = cpp_llm_engine::RealDeepSeek2ExecutorFactory::load(
        argv[1], &error);
    if (factory == nullptr) {
        std::cerr << "DeepSeek-V2-Lite load failed: " << error << '\n';
        return 1;
    }
    factory->set_layer_cache_capacity(factory->layer_count());

    std::vector<double> row_wise_samples;
    std::vector<double> grouped_samples;
    float max_error = 0.0F;
    double min_cosine = 1.0;
    bool tokens_match = true;
    std::size_t request_seed = 1;
    for (std::size_t iteration = 0; iteration < iterations; ++iteration) {
        Result first;
        Result second;
        const bool grouped_first = iteration % 2 == 1;
        const auto first_mode = grouped_first
            ? cpp_llm_engine::RealDeepSeek2MoeBatchMode::Grouped
            : cpp_llm_engine::RealDeepSeek2MoeBatchMode::RowWise;
        const auto second_mode = grouped_first
            ? cpp_llm_engine::RealDeepSeek2MoeBatchMode::RowWise
            : cpp_llm_engine::RealDeepSeek2MoeBatchMode::Grouped;
        if (!run(*factory, first_mode, request_seed, &first, &error) ||
            !run(*factory, second_mode, request_seed + 2, &second, &error)) {
            std::cerr << "DeepSeek-V2-Lite runtime A/B failed: "
                      << error << '\n';
            return 1;
        }
        request_seed += 4;
        const auto& row_wise = grouped_first ? second : first;
        const auto& grouped = grouped_first ? first : second;
        row_wise_samples.push_back(row_wise.milliseconds);
        grouped_samples.push_back(grouped.milliseconds);
        max_error = std::max(
            max_error, maximum_error(row_wise.logits, grouped.logits));
        min_cosine = std::min(
            min_cosine,
            minimum_cosine_similarity(row_wise.logits, grouped.logits));
        tokens_match = tokens_match && row_wise.tokens == grouped.tokens;
        std::cout << "paired_sample iteration=" << iteration + 1
                  << " path_order="
                  << (grouped_first ? "grouped-first" : "row-wise-first")
                  << " row_wise_ms=" << row_wise.milliseconds
                  << " grouped_ms=" << grouped.milliseconds << '\n';
    }

    const double row_wise_median = median(row_wise_samples);
    const double grouped_median = median(grouped_samples);
    const bool correct = tokens_match && max_error <= 5.0e-4F &&
        min_cosine >= 0.999999;
    std::cout << "iterations: " << iterations
              << "\nbatch: 2"
              << "\ncache_layers: " << factory->layer_count()
              << "\nrow_wise_median_ms: " << row_wise_median
              << "\ngrouped_median_ms: " << grouped_median
              << "\ngrouped_to_row_wise_ratio: "
              << grouped_median / row_wise_median
              << "\nmaximum_logits_error: " << max_error
              << "\nminimum_logits_cosine: " << min_cosine
              << "\ntokens_match: " << (tokens_match ? "true" : "false")
              << '\n';
    print_samples("row_wise", row_wise_samples);
    print_samples("grouped", grouped_samples);
    std::cout << "correctness: " << (correct ? "PASS" : "FAIL") << '\n';
    return correct ? 0 : 1;
}
