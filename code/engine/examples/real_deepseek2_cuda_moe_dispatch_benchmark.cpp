// Chapter 09: compare row-copy and packed CUDA MoE Prefill dispatch.
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
using cpp_llm_engine::model::CudaMoePrefillPath;

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
    CudaMoePrefillPath path,
    Result* result,
    std::string* error) {
    auto executor = factory.create_executor(
        cpp_llm_engine::cuda::CudaMlaComputePath::AbsorbedThreeStage, path);
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
    if (argc < 2 || argc > 5) {
        std::cerr << "usage: real_deepseek2_cuda_moe_dispatch_benchmark "
                     "<model.gguf> [lengths=32,128] [warmup=1] [iterations=4]\n";
        return 2;
    }
    const std::vector<std::size_t> lengths =
        parse_lengths(argc >= 3 ? argv[2] : "32,128");
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

    std::string error;
    auto factory = cpp_llm_engine::RealDeepSeek2CudaExecutorFactory::load(
        argv[1], *std::max_element(lengths.begin(), lengths.end()), &error);
    if (factory == nullptr) {
        std::cerr << "DeepSeek MoE dispatch setup failed: " << error << '\n';
        return 1;
    }

    std::cout << std::setprecision(9);
    bool all_correct = true;
    for (const std::size_t length : lengths) {
        const auto prompt = make_prompt(length);
        for (std::size_t index = 0; index < warmup; ++index) {
            Result row_copies;
            Result packed;
            if (!run_path(*factory, prompt, CudaMoePrefillPath::RowCopies,
                    &row_copies, &error) ||
                !run_path(*factory, prompt, CudaMoePrefillPath::PackedDispatch,
                    &packed, &error)) {
                std::cerr << "DeepSeek MoE dispatch warmup failed: "
                          << error << '\n';
                return 1;
            }
        }

        std::vector<double> row_samples;
        std::vector<double> packed_samples;
        float max_error = 0.0F;
        bool tokens_match = true;
        for (std::size_t iteration = 0; iteration < iterations; ++iteration) {
            const bool packed_first = iteration % 2 == 1;
            Result first;
            Result second;
            const auto first_path = packed_first
                ? CudaMoePrefillPath::PackedDispatch
                : CudaMoePrefillPath::RowCopies;
            const auto second_path = packed_first
                ? CudaMoePrefillPath::RowCopies
                : CudaMoePrefillPath::PackedDispatch;
            if (!run_path(*factory, prompt, first_path, &first, &error) ||
                !run_path(*factory, prompt, second_path, &second, &error)) {
                std::cerr << "DeepSeek MoE dispatch A/B failed: "
                          << error << '\n';
                return 1;
            }
            const Result& row_copies = packed_first ? second : first;
            const Result& packed = packed_first ? first : second;
            row_samples.push_back(row_copies.elapsed_ms);
            packed_samples.push_back(packed.elapsed_ms);
            max_error = std::max(
                max_error, maximum_error(row_copies.logits, packed.logits));
            tokens_match = tokens_match &&
                row_copies.greedy_token == packed.greedy_token;
            std::cout << "paired_sample length=" << length
                      << " iteration=" << iteration + 1
                      << " path_order="
                      << (packed_first ? "packed-first" : "row-copies-first")
                      << " row_copies_ms=" << row_copies.elapsed_ms
                      << " packed_dispatch_ms=" << packed.elapsed_ms << '\n';
        }
        const double row_ms = median(row_samples);
        const double packed_ms = median(packed_samples);
        const bool correct = tokens_match && max_error <= 1.0e-4F;
        all_correct = all_correct && correct;
        std::cout << "matrix_cell length=" << length
                  << " row_copies_median_ms=" << row_ms
                  << " packed_dispatch_median_ms=" << packed_ms
                  << " packed_to_row_copies_ratio=" << packed_ms / row_ms
                  << " maximum_logits_error=" << max_error
                  << " tokens_match=" << (tokens_match ? "true" : "false")
                  << " correctness=" << (correct ? "PASS" : "FAIL") << '\n';
    }
    std::cout << "moe_dispatch_correctness="
              << (all_correct ? "PASS" : "FAIL") << '\n';
    return all_correct ? 0 : 1;
}
