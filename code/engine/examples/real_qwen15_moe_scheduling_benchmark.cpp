// Chapter 08/09：比较 Qwen1.5-MoE 固定请求队列的逐行与分组调度执行。
// 计时边界：Prefill 在外，包含后续调度计划、批量 Decode 和请求提交。
#include "cpp_llm_engine/real_qwen2_moe_executor_factory.h"

#include "chapter06_generation/request.h"
#include "chapter06_generation/runtime.h"
#include "chapter06_generation/sampler.h"
#include "chapter08_scheduling/step_scheduler.h"

#include <algorithm>
#include <chrono>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

struct Result {
    double milliseconds = 0.0;
    std::vector<chapter06_generation::TokenSequence> generated;
    std::size_t decode_steps = 0;
    std::size_t model_batch_calls = 0;
    std::size_t tensor_batch_requests = 0;
};

bool run(cpp_llm_engine::RealQwen2MoeExecutorFactory& factory,
         cpp_llm_engine::RealQwen2MoeBatchMode mode,
         std::size_t request_count,
         std::size_t new_tokens,
         std::size_t request_seed,
         Result* result,
         std::string* error) {
    factory.set_moe_batch_mode(mode);
    chapter06_generation::GenerationConfig config;
    config.max_new_tokens = new_tokens;
    config.eos_token_id = factory.eos_token_id();
    chapter06_generation::Runtime runtime(config);
    chapter06_generation::GreedySampler sampler;
    chapter08_scheduling::ContinuousBatchScheduler scheduler;

    std::vector<std::unique_ptr<chapter06_generation::Request>> requests;
    std::vector<std::shared_ptr<chapter06_generation::RequestModelExecutor>>
        executors;
    std::vector<chapter08_scheduling::ScheduledRequest> scheduled;
    requests.reserve(request_count);
    executors.reserve(request_count);
    scheduled.reserve(request_count);
    for (std::size_t index = 0; index < request_count; ++index) {
        requests.push_back(
            std::make_unique<chapter06_generation::Request>(
                chapter06_generation::TokenSequence{0}));
        executors.push_back(factory.create_executor());
        if (executors.back() == nullptr) {
            if (error != nullptr) *error = "failed to create MoE executor";
            return false;
        }
        scheduled.push_back(
            {request_seed + index, requests.back().get(), &sampler,
             executors.back()});
    }

    const auto prefill = scheduler.run_step(
        runtime, scheduled, request_count, request_count, {}, nullptr, 0,
        nullptr, &factory);
    if (prefill.prefill_requests != request_count ||
        prefill.generated_tokens != request_count) {
        if (error != nullptr) *error = "MoE scheduling Prefill is incomplete";
        return false;
    }

    const auto start = Clock::now();
    while (std::any_of(
        requests.begin(), requests.end(), [](const auto& request) {
            return request != nullptr && !request->terminal();
        })) {
        const auto stats = scheduler.run_step(
            runtime, scheduled, request_count, request_count, {}, nullptr, 0,
            nullptr, &factory);
        if (stats.generated_tokens == 0) {
            if (error != nullptr) *error = "MoE scheduling made no progress";
            return false;
        }
        ++result->decode_steps;
        result->model_batch_calls += stats.model_batch_calls;
        result->tensor_batch_requests += stats.tensor_batch_requests;
        if (result->decode_steps > new_tokens) {
            if (error != nullptr) *error = "MoE scheduling exceeded step bound";
            return false;
        }
    }
    const auto stop = Clock::now();
    result->milliseconds =
        std::chrono::duration<double, std::milli>(stop - start).count();
    for (const auto& request : requests) {
        result->generated.push_back(request->generated());
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

void print_samples(const char* name, const std::vector<double>& samples) {
    std::cout << name << "_samples_ms:";
    for (const double sample : samples) std::cout << ' ' << sample;
    std::cout << '\n';
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2 || argc > 5) {
        std::cerr << "usage: real_qwen15_moe_scheduling_benchmark "
                     "<model.gguf> [requests=4] [new_tokens=4] "
                     "[iterations=2]\n";
        return 2;
    }
    const std::size_t request_count = argc >= 3
        ? static_cast<std::size_t>(std::stoul(argv[2])) : 4;
    const std::size_t new_tokens = argc >= 4
        ? static_cast<std::size_t>(std::stoul(argv[3])) : 4;
    const std::size_t iterations = argc >= 5
        ? static_cast<std::size_t>(std::stoul(argv[4])) : 2;
    if (request_count == 0 || new_tokens < 2 || iterations == 0) {
        std::cerr << "requests and iterations must be positive; new_tokens >= 2\n";
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

    std::vector<double> row_wise_samples;
    std::vector<double> grouped_samples;
    bool outputs_match = true;
    bool execution_shape_match = true;
    std::size_t request_seed = 1;
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
        if (!run(*factory, first_mode, request_count, new_tokens,
                 request_seed, &first, &error)) {
            std::cerr << "first scheduling path failed: " << error << '\n';
            return 1;
        }
        request_seed += request_count;
        if (!run(*factory, second_mode, request_count, new_tokens,
                 request_seed, &second, &error)) {
            std::cerr << "second scheduling path failed: " << error << '\n';
            return 1;
        }
        request_seed += request_count;
        const auto& row_wise = grouped_first ? second : first;
        const auto& grouped = grouped_first ? first : second;
        row_wise_samples.push_back(row_wise.milliseconds);
        grouped_samples.push_back(grouped.milliseconds);
        outputs_match = outputs_match &&
            row_wise.generated == grouped.generated;
        execution_shape_match = execution_shape_match &&
            row_wise.decode_steps == new_tokens - 1 &&
            grouped.decode_steps == new_tokens - 1 &&
            row_wise.model_batch_calls == new_tokens - 1 &&
            grouped.model_batch_calls == new_tokens - 1 &&
            row_wise.tensor_batch_requests ==
                request_count * (new_tokens - 1) &&
            grouped.tensor_batch_requests ==
                request_count * (new_tokens - 1);
        std::cout << "paired_iteration=" << iteration + 1
                  << " path_order="
                  << (grouped_first ? "grouped-first" : "row-wise-first")
                  << " row_wise_ms=" << row_wise.milliseconds
                  << " grouped_ms=" << grouped.milliseconds << '\n';
    }

    const double row_wise_median = median(row_wise_samples);
    const double grouped_median = median(grouped_samples);
    const bool correct = outputs_match && execution_shape_match;
    std::cout << "requests: " << request_count << '\n'
              << "new_tokens: " << new_tokens << '\n'
              << "timed_decode_steps: " << new_tokens - 1 << '\n'
              << "iterations: " << iterations << '\n'
              << "row_wise_median_ms: " << row_wise_median << '\n'
              << "grouped_median_ms: " << grouped_median << '\n'
              << "grouped_to_row_wise_ratio: "
              << grouped_median / row_wise_median << '\n'
              << "outputs_match: " << (outputs_match ? "true" : "false")
              << '\n'
              << "execution_shape_match: "
              << (execution_shape_match ? "true" : "false") << '\n';
    print_samples("row_wise", row_wise_samples);
    print_samples("grouped", grouped_samples);
    std::cout << "correctness: " << (correct ? "PASS" : "FAIL") << '\n';
    return correct ? 0 : 1;
}
