// Chapter 09: compare continuous and paged DeepSeek2 CUDA MLA.
// Timing excludes model loading, context creation, and terminal release.
#include "cpp_llm_engine/real_deepseek2_cuda_executor_factory.h"

#include "chapter06_generation/sampler.h"
#include "chapter07_kv_cache/block_table.h"

#include <cuda_runtime_api.h>

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
    double prefill_ms = 0.0;
    double decode_ms = 0.0;
    chapter06_generation::Logits prefill_logits;
    chapter06_generation::Logits decode_logits;
    chapter06_generation::TokenId decode_token = 0;
    chapter06_generation::TokenId next_token = 0;
};

bool run_path(
    cpp_llm_engine::RealDeepSeek2CudaExecutorFactory& factory,
    bool paged,
    std::size_t request_id,
    Result* result,
    std::string* error) {
    auto executor = factory.create_executor();
    if (executor == nullptr || result == nullptr) {
        if (error != nullptr) *error = "failed to create benchmark executor";
        return false;
    }
    if (paged && !factory.bind_request(executor, request_id, error)) {
        return false;
    }

    const auto prefill_start = Clock::now();
    const bool prefilled = executor->prefill({11, 207}, &result->prefill_logits, error);
    const auto prefill_stop = Clock::now();
    if (!prefilled) {
        if (paged) factory.release_request(request_id);
        return false;
    }
    result->prefill_ms = std::chrono::duration<double, std::milli>(
        prefill_stop - prefill_start).count();

    chapter06_generation::GreedySampler sampler;
    result->decode_token = sampler.sample(result->prefill_logits);
    const auto decode_start = Clock::now();
    const bool decoded = executor->decode(
        result->decode_token, &result->decode_logits, error);
    const auto decode_stop = Clock::now();
    if (!decoded) {
        if (paged) factory.release_request(request_id);
        return false;
    }
    result->decode_ms = std::chrono::duration<double, std::milli>(
        decode_stop - decode_start).count();
    result->next_token = sampler.sample(result->decode_logits);

    const bool position_ok =
        executor->execution_state().committed_tokens == 3;
    if (paged) factory.release_request(request_id);
    return position_ok;
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
        maximum = std::max(
            maximum, std::fabs(left[index] - right[index]));
    }
    return maximum;
}

void print_samples(const char* name, const std::vector<double>& samples) {
    std::cout << name << "_samples_ms:";
    for (const double sample : samples) std::cout << ' ' << sample;
    std::cout << '\n';
}

}  // namespace

int main(int argc, char** argv) {
    std::cout << std::setprecision(9);
    if (argc < 2 || argc > 4) {
        std::cerr << "usage: real_deepseek2_cuda_paged_mla_benchmark "
                     "<model.gguf> [warmup=1] [iterations=4]\n";
        return 2;
    }
    const std::size_t warmup = argc >= 3
        ? static_cast<std::size_t>(std::stoul(argv[2])) : 1;
    const std::size_t iterations = argc == 4
        ? static_cast<std::size_t>(std::stoul(argv[3])) : 4;
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
    if (factory == nullptr ||
        !factory->configure_paged_mla_cache(128, 2, &error)) {
        std::cerr << "DeepSeek paged MLA benchmark setup failed: "
                  << error << '\n';
        return 1;
    }
    const auto resource = factory->paged_mla_resource();

    std::size_t request_id = 10000;
    for (std::size_t index = 0; index < warmup; ++index) {
        Result continuous;
        Result paged;
        if (!run_path(*factory, false, request_id++, &continuous, &error) ||
            !run_path(*factory, true, request_id++, &paged, &error)) {
            std::cerr << "DeepSeek paged MLA warmup failed: " << error << '\n';
            return 1;
        }
    }

    std::vector<double> continuous_prefill_samples;
    std::vector<double> paged_prefill_samples;
    std::vector<double> continuous_decode_samples;
    std::vector<double> paged_decode_samples;
    float prefill_max_error = 0.0F;
    float decode_max_error = 0.0F;
    bool tokens_match = true;
    for (std::size_t iteration = 0; iteration < iterations; ++iteration) {
        Result first;
        Result second;
        const bool paged_first = iteration % 2 == 1;
        const bool first_ok = run_path(
            *factory, paged_first, request_id++, &first, &error);
        const bool second_ok = run_path(
            *factory, !paged_first, request_id++, &second, &error);
        if (!first_ok || !second_ok) {
            std::cerr << "DeepSeek paged MLA A/B failed: " << error << '\n';
            return 1;
        }
        const Result& continuous = paged_first ? second : first;
        const Result& paged = paged_first ? first : second;
        continuous_prefill_samples.push_back(continuous.prefill_ms);
        paged_prefill_samples.push_back(paged.prefill_ms);
        continuous_decode_samples.push_back(continuous.decode_ms);
        paged_decode_samples.push_back(paged.decode_ms);
        prefill_max_error = std::max(
            prefill_max_error,
            maximum_error(continuous.prefill_logits, paged.prefill_logits));
        decode_max_error = std::max(
            decode_max_error,
            maximum_error(continuous.decode_logits, paged.decode_logits));
        tokens_match = tokens_match &&
            continuous.decode_token == paged.decode_token &&
            continuous.next_token == paged.next_token;
        std::cout << "paired_sample iteration=" << iteration + 1
                  << " path_order="
                  << (paged_first ? "paged-first" : "continuous-first")
                  << " continuous_prefill_ms=" << continuous.prefill_ms
                  << " paged_prefill_ms=" << paged.prefill_ms
                  << " continuous_decode_ms=" << continuous.decode_ms
                  << " paged_decode_ms=" << paged.decode_ms << '\n';
    }

    const double continuous_prefill = median(continuous_prefill_samples);
    const double paged_prefill = median(paged_prefill_samples);
    const double continuous_decode = median(continuous_decode_samples);
    const double paged_decode = median(paged_decode_samples);
    const bool released = resource != nullptr && resource->used_blocks() == 0;
    const bool correct = tokens_match && released &&
        prefill_max_error <= 1.0e-4F && decode_max_error <= 1.0e-4F;
    std::cout << "warmup: " << warmup
              << "\niterations: " << iterations
              << "\nprompt_tokens: 2"
              << "\ndecode_steps: 1"
              << "\ncontinuous_prefill_median_ms: " << continuous_prefill
              << "\npaged_prefill_median_ms: " << paged_prefill
              << "\npaged_to_continuous_prefill_ratio: "
              << paged_prefill / continuous_prefill
              << "\ncontinuous_decode_median_ms: " << continuous_decode
              << "\npaged_decode_median_ms: " << paged_decode
              << "\npaged_to_continuous_decode_ratio: "
              << paged_decode / continuous_decode
              << "\nprefill_maximum_logits_error: " << prefill_max_error
              << "\ndecode_maximum_logits_error: " << decode_max_error
              << "\ntokens_match: " << (tokens_match ? "true" : "false")
              << "\nterminal_blocks_released: "
              << (released ? "true" : "false") << '\n';
    print_samples("continuous_prefill", continuous_prefill_samples);
    print_samples("paged_prefill", paged_prefill_samples);
    print_samples("continuous_decode", continuous_decode_samples);
    print_samples("paged_decode", paged_decode_samples);
    std::cout << "correctness: " << (correct ? "PASS" : "FAIL") << '\n';
    return correct ? 0 : 1;
}
