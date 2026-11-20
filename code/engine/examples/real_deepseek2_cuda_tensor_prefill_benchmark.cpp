// Chapter 09: compare sequential and Tensor Prefill for DeepSeek2 CUDA.
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
    bool tensor_path,
    Result* result,
    std::string* error) {
    auto executor = factory.create_executor();
    if (executor == nullptr || result == nullptr) {
        if (error != nullptr) *error = "failed to create prefill executor";
        return false;
    }

    const auto start = Clock::now();
    bool succeeded = false;
    if (tensor_path) {
        succeeded = executor->prefill(prompt, &result->logits, error);
    } else {
        for (std::size_t index = 0; index < prompt.size(); ++index) {
            succeeded = executor->prefill_chunk(
                {prompt[index]},
                index == 0,
                index + 1 == prompt.size(),
                index + 1 == prompt.size() ? &result->logits : nullptr,
                error);
            if (!succeeded) break;
        }
    }
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

void print_samples(const char* name, const std::vector<double>& samples) {
    std::cout << name << '_';
    for (std::size_t index = 0; index < samples.size(); ++index) {
        if (index > 0) std::cout << ',';
        std::cout << samples[index];
    }
}

}  // namespace

int main(int argc, char** argv) {
    std::cout << std::setprecision(9);
    if (argc < 2 || argc > 5) {
        std::cerr << "usage: real_deepseek2_cuda_tensor_prefill_benchmark "
                     "<model.gguf> [lengths=2,8,32,128] [warmup=1] "
                     "[iterations=4]\n";
        return 2;
    }
    const std::vector<std::size_t> lengths = parse_lengths(
        argc >= 3 ? argv[2] : "2,8,32,128");
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
        std::cerr << "DeepSeek Tensor Prefill setup failed: " << error << '\n';
        return 1;
    }

    bool all_correct = true;
    for (const std::size_t length : lengths) {
        const auto prompt = make_prompt(length);
        for (std::size_t index = 0; index < warmup; ++index) {
            Result sequential;
            Result tensor;
            if (!run_path(*factory, prompt, false, &sequential, &error) ||
                !run_path(*factory, prompt, true, &tensor, &error)) {
                std::cerr << "DeepSeek Tensor Prefill warmup failed: "
                          << error << '\n';
                return 1;
            }
        }

        std::vector<double> sequential_samples;
        std::vector<double> tensor_samples;
        float max_error = 0.0F;
        bool tokens_match = true;
        for (std::size_t iteration = 0; iteration < iterations; ++iteration) {
            const bool tensor_first = iteration % 2 == 1;
            Result first;
            Result second;
            if (!run_path(
                    *factory, prompt, tensor_first, &first, &error) ||
                !run_path(
                    *factory, prompt, !tensor_first, &second, &error)) {
                std::cerr << "DeepSeek Tensor Prefill A/B failed: "
                          << error << '\n';
                return 1;
            }
            const Result& sequential = tensor_first ? second : first;
            const Result& tensor = tensor_first ? first : second;
            sequential_samples.push_back(sequential.elapsed_ms);
            tensor_samples.push_back(tensor.elapsed_ms);
            max_error = std::max(
                max_error, maximum_error(sequential.logits, tensor.logits));
            tokens_match = tokens_match &&
                sequential.greedy_token == tensor.greedy_token;
            std::cout << "paired_sample length=" << length
                      << " iteration=" << iteration + 1
                      << " path_order="
                      << (tensor_first ? "tensor-first" : "sequential-first")
                      << " sequential_ms=" << sequential.elapsed_ms
                      << " tensor_ms=" << tensor.elapsed_ms << '\n';
        }

        const double sequential_ms = median(sequential_samples);
        const double tensor_ms = median(tensor_samples);
        const bool correct = tokens_match && max_error <= 1.0e-4F;
        all_correct = all_correct && correct;
        std::cout << "matrix_cell length=" << length
                  << " sequential_median_ms=" << sequential_ms
                  << " tensor_median_ms=" << tensor_ms
                  << " tensor_to_sequential_ratio=" << tensor_ms / sequential_ms
                  << " maximum_logits_error=" << max_error
                  << " tokens_match=" << (tokens_match ? "true" : "false")
                  << ' ';
        print_samples("sequential_samples_ms", sequential_samples);
        std::cout << ' ';
        print_samples("tensor_samples_ms", tensor_samples);
        std::cout << " correctness=" << (correct ? "PASS" : "FAIL") << '\n';
    }
    std::cout << "tensor_prefill_correctness="
              << (all_correct ? "PASS" : "FAIL") << '\n';
    return all_correct ? 0 : 1;
}
