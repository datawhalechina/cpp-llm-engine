// Chapter 09: compare sequential and grouped DeepSeek2 CUDA Decode.
// Timing excludes model loading, context creation, and Prefill.
#include "cpp_llm_engine/real_deepseek2_cuda_executor_factory.h"

#include "chapter06_generation/sampler.h"

#include <cuda_runtime_api.h>
#include <cuda_profiler_api.h>

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

bool prepare(
    cpp_llm_engine::RealDeepSeek2CudaExecutorFactory& factory,
    std::vector<std::shared_ptr<chapter06_generation::RequestModelExecutor>>*
        executors,
    chapter06_generation::TokenSequence* decode_tokens,
    std::string* error) {
    chapter06_generation::GreedySampler sampler;
    for (std::size_t request = 0; request < 2; ++request) {
        auto executor = factory.create_executor();
        chapter06_generation::Logits logits;
        if (executor == nullptr || !executor->prefill({11}, &logits, error)) {
            return false;
        }
        decode_tokens->push_back(sampler.sample(logits));
        executors->push_back(std::move(executor));
    }
    if (*decode_tokens != chapter06_generation::TokenSequence{207, 207}) {
        if (error != nullptr) *error = "unexpected Prefill token";
        return false;
    }
    return true;
}

bool run_sequential(
    cpp_llm_engine::RealDeepSeek2CudaExecutorFactory& factory,
    Result* result,
    std::string* error) {
    std::vector<std::shared_ptr<chapter06_generation::RequestModelExecutor>>
        executors;
    chapter06_generation::TokenSequence decode_tokens;
    if (!prepare(factory, &executors, &decode_tokens, error)) return false;

    result->logits.resize(2);
    const auto start = Clock::now();
    for (std::size_t request = 0; request < 2; ++request) {
        if (!executors[request]->decode(
                decode_tokens[request], &result->logits[request], error)) {
            return false;
        }
    }
    const auto stop = Clock::now();
    result->milliseconds =
        std::chrono::duration<double, std::milli>(stop - start).count();

    chapter06_generation::GreedySampler sampler;
    for (const auto& row : result->logits) {
        result->tokens.push_back(sampler.sample(row));
    }
    return executors[0]->execution_state().committed_tokens == 2 &&
        executors[1]->execution_state().committed_tokens == 2;
}

bool run_grouped(
    cpp_llm_engine::RealDeepSeek2CudaExecutorFactory& factory,
    std::size_t request_seed,
    Result* result,
    std::string* error,
    bool profiler_range = false) {
    std::vector<std::shared_ptr<chapter06_generation::RequestModelExecutor>>
        executors;
    chapter06_generation::TokenSequence decode_tokens;
    if (!prepare(factory, &executors, &decode_tokens, error)) return false;

    if (profiler_range && cudaProfilerStart() != cudaSuccess) {
        if (error != nullptr) *error = "cudaProfilerStart failed";
        return false;
    }
    const auto start = Clock::now();
    const bool decoded = factory.decode_batch(
            executors, decode_tokens, {1, 1},
            {request_seed, request_seed + 1}, &result->logits, error);
    const auto stop = Clock::now();
    if (profiler_range && cudaProfilerStop() != cudaSuccess) {
        if (error != nullptr) *error = "cudaProfilerStop failed";
        return false;
    }
    if (!decoded) return false;
    result->milliseconds =
        std::chrono::duration<double, std::milli>(stop - start).count();

    chapter06_generation::GreedySampler sampler;
    for (const auto& row : result->logits) {
        result->tokens.push_back(sampler.sample(row));
    }
    return executors[0]->execution_state().committed_tokens == 2 &&
        executors[1]->execution_state().committed_tokens == 2;
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
        std::cerr << "usage: real_deepseek2_cuda_grouped_moe_benchmark "
                     "<model.gguf> [iterations=2|--profile-grouped]\n";
        return 2;
    }
    const bool profile_grouped = argc == 3 &&
        std::string(argv[2]) == "--profile-grouped";
    const std::size_t iterations = argc == 3 && !profile_grouped
        ? static_cast<std::size_t>(std::stoul(argv[2])) : 2;
    if (iterations < 2) {
        std::cerr << "iterations must be at least 2 for reversed order\n";
        return 2;
    }
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        std::cerr << "CUDA device is unavailable\n";
        return 1;
    }

    std::string error;
    auto factory = cpp_llm_engine::RealDeepSeek2CudaExecutorFactory::load(
        argv[1], 8, &error);
    if (factory == nullptr) {
        std::cerr << "DeepSeek-V2-Lite load failed: " << error << '\n';
        return 1;
    }

    if (profile_grouped) {
        Result grouped;
        if (!run_grouped(*factory, 900, &grouped, &error, true)) {
            std::cerr << "DeepSeek CUDA grouped profile failed: "
                      << error << '\n';
            return 1;
        }
        const bool correct = grouped.logits.size() == 2 &&
            grouped.tokens.size() == 2 &&
            grouped.tokens[0] == grouped.tokens[1] &&
            factory->grouped_moe_call_count(2) == 26;
        std::cout << "profile_grouped_ms: " << grouped.milliseconds
                  << "\ngrouped_moe_calls: "
                  << factory->grouped_moe_call_count(2)
                  << "\nactive_expert_groups: "
                  << factory->active_expert_group_count(2)
                  << "\nmulti_row_expert_groups: "
                  << factory->multi_row_expert_group_count(2)
                  << "\nprofile_correctness: "
                  << (correct ? "PASS" : "FAIL") << '\n';
        return correct ? 0 : 1;
    }

    std::vector<double> sequential_samples;
    std::vector<double> grouped_samples;
    float max_error = 0.0F;
    double min_cosine = 1.0;
    bool tokens_match = true;
    const std::size_t grouped_uses_before =
        factory->batch_workspace_use_count(2);
    const std::size_t grouped_calls_before =
        factory->grouped_moe_call_count(2);
    const std::size_t active_groups_before =
        factory->active_expert_group_count(2);
    const std::size_t multi_row_groups_before =
        factory->multi_row_expert_group_count(2);
    const std::size_t routed_rows_before = factory->routed_row_count(2);
    for (std::size_t iteration = 0; iteration < iterations; ++iteration) {
        Result first;
        Result second;
        const bool grouped_first = iteration % 2 == 1;
        const bool first_ok = grouped_first
            ? run_grouped(*factory, 100 + iteration * 4, &first, &error)
            : run_sequential(*factory, &first, &error);
        const bool second_ok = grouped_first
            ? run_sequential(*factory, &second, &error)
            : run_grouped(*factory, 102 + iteration * 4, &second, &error);
        if (!first_ok || !second_ok) {
            std::cerr << "DeepSeek CUDA grouped MoE A/B failed: "
                      << error << '\n';
            return 1;
        }
        const Result& sequential = grouped_first ? second : first;
        const Result& grouped = grouped_first ? first : second;
        sequential_samples.push_back(sequential.milliseconds);
        grouped_samples.push_back(grouped.milliseconds);
        max_error = std::max(
            max_error, maximum_error(sequential.logits, grouped.logits));
        min_cosine = std::min(
            min_cosine,
            minimum_cosine_similarity(sequential.logits, grouped.logits));
        tokens_match = tokens_match && sequential.tokens == grouped.tokens;
        std::cout << "paired_sample iteration=" << iteration + 1
                  << " path_order="
                  << (grouped_first ? "grouped-first" : "sequential-first")
                  << " sequential_ms=" << sequential.milliseconds
                  << " grouped_ms=" << grouped.milliseconds << '\n';
    }

    const std::size_t grouped_uses =
        factory->batch_workspace_use_count(2) - grouped_uses_before;
    const std::size_t grouped_calls =
        factory->grouped_moe_call_count(2) - grouped_calls_before;
    const std::size_t active_groups =
        factory->active_expert_group_count(2) - active_groups_before;
    const std::size_t multi_row_groups =
        factory->multi_row_expert_group_count(2) - multi_row_groups_before;
    const std::size_t routed_rows =
        factory->routed_row_count(2) - routed_rows_before;
    const double sequential_median = median(sequential_samples);
    const double grouped_median = median(grouped_samples);
    const bool correct = tokens_match && max_error <= 5.0e-4F &&
        min_cosine >= 0.999999 &&
        grouped_uses == iterations && grouped_calls == iterations * 26;
    std::cout << "iterations: " << iterations
              << "\nbatch: 2"
              << "\nsequential_median_ms: " << sequential_median
              << "\ngrouped_median_ms: " << grouped_median
              << "\ngrouped_to_sequential_ratio: "
              << grouped_median / sequential_median
              << "\nmaximum_logits_error: " << max_error
              << "\nminimum_logits_cosine: " << min_cosine
              << "\ntokens_match: " << (tokens_match ? "true" : "false")
              << "\nbatch_workspace_uses: " << grouped_uses
              << "\ngrouped_moe_calls: " << grouped_calls
              << "\nactive_expert_groups: " << active_groups
              << "\nmulti_row_expert_groups: " << multi_row_groups
              << "\nrouted_rows: " << routed_rows
              << "\nmulti_row_group_ratio: "
              << (active_groups == 0 ? 0.0 :
                  static_cast<double>(multi_row_groups) / active_groups)
              << '\n';
    print_samples("sequential", sequential_samples);
    print_samples("grouped", grouped_samples);
    std::cout << "correctness: " << (correct ? "PASS" : "FAIL") << '\n';
    return correct ? 0 : 1;
}
