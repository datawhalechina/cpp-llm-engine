// Chapter 09: compare quantized GEMV paths in real DeepSeek Prefill.
// Timing excludes model loading and request-context creation.
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
using cpp_llm_engine::cuda::CudaQuantizedGemvPath;

struct Result final {
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
        const auto length = static_cast<std::size_t>(std::stoul(item));
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
    CudaQuantizedGemvPath path,
    Result* result,
    std::string* error) {
    auto executor = factory.create_executor(
        cpp_llm_engine::cuda::CudaMlaComputePath::AbsorbedThreeStage,
        cpp_llm_engine::model::CudaMoePrefillPath::PackedDispatch,
        path);
    if (executor == nullptr || result == nullptr) return false;
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
    if (argc < 2 || argc > 6) {
        std::cerr << "usage: real_deepseek2_cuda_iq4_warp_benchmark "
                     "<model.gguf> [lengths=32,128] [warmup=1] [iterations=4] "
                     "[comparison=iq4|q2|q3]\n";
        return 2;
    }
    const auto lengths = parse_lengths(argc >= 3 ? argv[2] : "32,128");
    const std::size_t warmup = argc >= 4
        ? static_cast<std::size_t>(std::stoul(argv[3])) : 1;
    const std::size_t iterations = argc == 5
        ? static_cast<std::size_t>(std::stoul(argv[4]))
        : argc == 6 ? static_cast<std::size_t>(std::stoul(argv[4])) : 4;
    const std::string comparison = argc == 6 ? argv[5] : "iq4";
    if (lengths.empty() || iterations < 2) {
        std::cerr << "lengths must be positive and iterations at least 2\n";
        return 2;
    }
    if (comparison != "iq4" && comparison != "q2" && comparison != "q3") {
        std::cerr << "comparison must be iq4, q2, or q3\n";
        return 2;
    }
    const auto baseline_path = comparison == "iq4"
        ? CudaQuantizedGemvPath::Adaptive
        : comparison == "q2"
            ? CudaQuantizedGemvPath::AdaptiveIq4Warp
            : CudaQuantizedGemvPath::AdaptiveBlockWarp;
    const auto experiment_path = comparison == "iq4"
        ? CudaQuantizedGemvPath::AdaptiveIq4Warp
        : comparison == "q2"
            ? CudaQuantizedGemvPath::AdaptiveBlockWarp
            : CudaQuantizedGemvPath::AdaptiveAllWarp;
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        std::cerr << "CUDA device is unavailable\n";
        return 1;
    }

    std::string error;
    auto factory = cpp_llm_engine::RealDeepSeek2CudaExecutorFactory::load(
        argv[1], *std::max_element(lengths.begin(), lengths.end()), &error);
    if (factory == nullptr) {
        std::cerr << "DeepSeek IQ4 warp setup failed: " << error << '\n';
        return 1;
    }

    std::cout << std::setprecision(9);
    bool all_correct = true;
    for (const std::size_t length : lengths) {
        const auto prompt = make_prompt(length);
        for (std::size_t index = 0; index < warmup; ++index) {
            Result baseline;
            Result warp;
            if (!run_path(*factory, prompt, baseline_path, &baseline, &error) ||
                !run_path(*factory, prompt, experiment_path, &warp, &error)) {
                std::cerr << "DeepSeek GEMV path warmup failed: " << error << '\n';
                return 1;
            }
        }

        std::vector<double> baseline_samples;
        std::vector<double> warp_samples;
        float max_error = 0.0F;
        bool tokens_match = true;
        for (std::size_t iteration = 0; iteration < iterations; ++iteration) {
            const bool warp_first = iteration % 2 == 1;
            Result first;
            Result second;
            const auto first_path = warp_first
                ? experiment_path
                : baseline_path;
            const auto second_path = warp_first
                ? baseline_path
                : experiment_path;
            if (!run_path(*factory, prompt, first_path, &first, &error) ||
                !run_path(*factory, prompt, second_path, &second, &error)) {
                std::cerr << "DeepSeek GEMV path A/B failed: " << error << '\n';
                return 1;
            }
            const Result& baseline = warp_first ? second : first;
            const Result& warp = warp_first ? first : second;
            baseline_samples.push_back(baseline.elapsed_ms);
            warp_samples.push_back(warp.elapsed_ms);
            max_error = std::max(
                max_error, maximum_error(baseline.logits, warp.logits));
            tokens_match = tokens_match &&
                baseline.greedy_token == warp.greedy_token;
            std::cout << "paired_sample length=" << length
                      << " iteration=" << iteration + 1
                      << " path_order="
                      << (warp_first ? "experiment-first" : "baseline-first")
                      << " comparison=" << comparison
                      << " baseline_ms=" << baseline.elapsed_ms
                      << " experiment_ms=" << warp.elapsed_ms << '\n';
        }
        const double baseline_ms = median(baseline_samples);
        const double warp_ms = median(warp_samples);
        const bool correct = tokens_match && max_error <= 1.0e-4F;
        all_correct = all_correct && correct;
        std::cout << "matrix_cell length=" << length
                  << " comparison=" << comparison
                  << " baseline_median_ms=" << baseline_ms
                  << " experiment_median_ms=" << warp_ms
                  << " experiment_to_baseline_ratio=" << warp_ms / baseline_ms
                  << " maximum_logits_error=" << max_error
                  << " tokens_match=" << (tokens_match ? "true" : "false")
                  << " correctness=" << (correct ? "PASS" : "FAIL") << '\n';
    }
    std::cout << "gemv_path_correctness="
              << (all_correct ? "PASS" : "FAIL") << '\n';
    return all_correct ? 0 : 1;
}
