// Chapter 09: compare legacy and weight-absorbed MLA in real CUDA Prefill.
#include "cpp_llm_engine/real_deepseek2_cuda_executor_factory.h"

#include "chapter06_generation/sampler.h"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;
using cpp_llm_engine::cuda::CudaMlaComputePath;

struct Result {
    double elapsed_ms = 0.0;
    chapter06_generation::Logits logits;
    chapter06_generation::TokenId greedy_token = 0;
};

std::vector<std::size_t> parse_lengths(const std::string& text) {
    std::vector<std::size_t> lengths;
    std::istringstream input(text);
    std::string item;
    while (std::getline(input, item, ',')) {
        if (item.empty()) return {};
        const std::size_t length = static_cast<std::size_t>(std::stoul(item));
        if (length == 0) return {};
        lengths.push_back(length);
    }
    return lengths;
}

chapter06_generation::TokenSequence make_prompt(std::size_t length) {
    static const chapter06_generation::TokenId pattern[] = {11, 207, 185, 304};
    chapter06_generation::TokenSequence prompt;
    prompt.reserve(length);
    for (std::size_t index = 0; index < length; ++index) {
        prompt.push_back(pattern[index % 4]);
    }
    return prompt;
}

bool run_path(
    cpp_llm_engine::RealDeepSeek2CudaExecutorFactory& factory,
    const chapter06_generation::TokenSequence& prompt,
    CudaMlaComputePath path,
    Result* result,
    std::string* error) {
    auto executor = factory.create_executor(path);
    if (executor == nullptr || result == nullptr) {
        if (error != nullptr) *error = "failed to create MLA A/B executor";
        return false;
    }
    const auto start = Clock::now();
    const bool succeeded = executor->prefill(prompt, &result->logits, error);
    const auto stop = Clock::now();
    if (!succeeded ||
        executor->execution_state().committed_tokens != prompt.size()) {
        return false;
    }
    result->elapsed_ms = std::chrono::duration<double, std::milli>(
        stop - start).count();
    result->greedy_token =
        chapter06_generation::GreedySampler{}.sample(result->logits);
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
    const chapter06_generation::Logits& left,
    const chapter06_generation::Logits& right) {
    if (left.size() != right.size()) return INFINITY;
    float maximum = 0.0F;
    for (std::size_t index = 0; index < left.size(); ++index) {
        maximum = std::max(maximum, std::fabs(left[index] - right[index]));
    }
    return maximum;
}

}  // namespace

int main(int argc, char** argv) {
    std::cout << std::setprecision(9);
    if (argc < 2 || argc > 5) {
        std::cerr << "usage: real_deepseek2_cuda_absorbed_mla_benchmark "
                     "<model.gguf> [lengths=32,128] [warmup=1] "
                     "[iterations=4]\n";
        return 2;
    }
    const auto lengths = parse_lengths(argc >= 3 ? argv[2] : "32,128");
    const std::size_t warmup = argc >= 4
        ? static_cast<std::size_t>(std::stoul(argv[3])) : 1;
    const std::size_t iterations = argc == 5
        ? static_cast<std::size_t>(std::stoul(argv[4])) : 4;
    if (lengths.empty() || iterations < 2) {
        std::cerr << "lengths must be positive and iterations at least 2\n";
        return 2;
    }
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        std::cerr << "CUDA device is unavailable\n";
        return 1;
    }

    const std::size_t capacity =
        *std::max_element(lengths.begin(), lengths.end());
    std::string error;
    auto factory = cpp_llm_engine::RealDeepSeek2CudaExecutorFactory::load(
        argv[1], capacity, &error);
    if (factory == nullptr) {
        std::cerr << "DeepSeek absorbed MLA setup failed: " << error << '\n';
        return 1;
    }

    bool all_correct = true;
    for (const std::size_t length : lengths) {
        const auto prompt = make_prompt(length);
        for (std::size_t index = 0; index < warmup; ++index) {
            Result legacy;
            Result absorbed;
            if (!run_path(*factory, prompt, CudaMlaComputePath::LegacyThreeStage,
                    &legacy, &error) ||
                !run_path(*factory, prompt, CudaMlaComputePath::AbsorbedThreeStage,
                    &absorbed, &error)) {
                std::cerr << "DeepSeek absorbed MLA warmup failed: "
                          << error << '\n';
                return 1;
            }
        }

        std::vector<double> legacy_samples;
        std::vector<double> absorbed_samples;
        float max_error = 0.0F;
        bool tokens_match = true;
        for (std::size_t iteration = 0; iteration < iterations; ++iteration) {
            const bool absorbed_first = iteration % 2 == 1;
            Result first;
            Result second;
            if (!run_path(*factory, prompt,
                    absorbed_first ? CudaMlaComputePath::AbsorbedThreeStage
                                   : CudaMlaComputePath::LegacyThreeStage,
                    &first, &error) ||
                !run_path(*factory, prompt,
                    absorbed_first ? CudaMlaComputePath::LegacyThreeStage
                                   : CudaMlaComputePath::AbsorbedThreeStage,
                    &second, &error)) {
                std::cerr << "DeepSeek absorbed MLA A/B failed: "
                          << error << '\n';
                return 1;
            }
            const Result& legacy = absorbed_first ? second : first;
            const Result& absorbed = absorbed_first ? first : second;
            legacy_samples.push_back(legacy.elapsed_ms);
            absorbed_samples.push_back(absorbed.elapsed_ms);
            max_error = std::max(
                max_error, maximum_error(legacy.logits, absorbed.logits));
            tokens_match = tokens_match &&
                legacy.greedy_token == absorbed.greedy_token;
            std::cout << "paired_sample length=" << length
                      << " iteration=" << iteration + 1
                      << " path_order="
                      << (absorbed_first ? "absorbed-first" : "legacy-first")
                      << " legacy_ms=" << legacy.elapsed_ms
                      << " absorbed_ms=" << absorbed.elapsed_ms << '\n';
        }
        const double legacy_ms = median(legacy_samples);
        const double absorbed_ms = median(absorbed_samples);
        const bool correct = tokens_match && max_error <= 1.0e-3F;
        all_correct = all_correct && correct;
        std::cout << "matrix_cell length=" << length
                  << " legacy_median_ms=" << legacy_ms
                  << " absorbed_median_ms=" << absorbed_ms
                  << " absorbed_to_legacy_ratio=" << absorbed_ms / legacy_ms
                  << " maximum_logits_error=" << max_error
                  << " tokens_match=" << (tokens_match ? "true" : "false")
                  << " correctness=" << (correct ? "PASS" : "FAIL") << '\n';
    }
    std::cout << "absorbed_mla_correctness="
              << (all_correct ? "PASS" : "FAIL") << '\n';
    return all_correct ? 0 : 1;
}
