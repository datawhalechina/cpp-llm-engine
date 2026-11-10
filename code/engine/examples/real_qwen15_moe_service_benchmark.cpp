// Chapter 10：比较动态 admission 下 Qwen1.5-MoE 逐行与分组服务执行。
#include "cpp_llm_engine/real_qwen2_moe_executor_factory.h"

#include "chapter06_generation/request.h"
#include "chapter06_generation/runtime.h"
#include "chapter06_generation/sampler.h"
#include "chapter10_serving/service_core.h"

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
    std::size_t model_batch_calls = 0;
    std::size_t tensor_batch_requests = 0;
    chapter10_serving::ServiceMetrics metrics;
};

bool run(cpp_llm_engine::RealQwen2MoeExecutorFactory& factory,
         cpp_llm_engine::RealQwen2MoeBatchMode mode,
         std::size_t seed,
         Result* result,
         std::string* error) {
    constexpr std::size_t kRequests = 4;
    constexpr std::size_t kNewTokens = 4;
    factory.set_moe_batch_mode(mode);
    chapter06_generation::GenerationConfig config;
    config.max_new_tokens = kNewTokens;
    config.eos_token_id = factory.eos_token_id();
    chapter06_generation::Runtime runtime(config);
    chapter06_generation::GreedySampler sampler;
    chapter10_serving::ServiceCore service(4, 4, {}, {}, {}, &factory);
    std::vector<std::unique_ptr<chapter06_generation::Request>> requests;
    requests.reserve(kRequests);
    for (std::size_t index = 0; index < kRequests; ++index) {
        requests.push_back(std::make_unique<chapter06_generation::Request>(
            chapter06_generation::TokenSequence{0}));
    }
    const auto admit = [&](std::size_t index) {
        return service.admit(seed + index, *requests[index], sampler,
                             factory.create_executor()).accepted();
    };
    if (!admit(0) || !admit(1)) {
        if (error != nullptr) *error = "initial service admission failed";
        return false;
    }
    const auto prefill = service.dispatch_step(runtime, 4);
    if (prefill.generated_tokens != 2) {
        if (error != nullptr) *error = "initial service Prefill failed";
        return false;
    }

    const auto start = Clock::now();
    if (!admit(2) || !admit(3)) {
        if (error != nullptr) *error = "dynamic service admission failed";
        return false;
    }
    std::size_t steps = 0;
    while (std::any_of(requests.begin(), requests.end(), [](const auto& request) {
        return request != nullptr && !request->terminal();
    })) {
        const auto stats = service.dispatch_step(runtime, 4);
        result->model_batch_calls += stats.model_batch_calls;
        result->tensor_batch_requests += stats.tensor_batch_requests;
        service.release_terminal();
        if (stats.generated_tokens == 0 || ++steps > 8) {
            if (error != nullptr) *error = "service dispatch made no progress";
            return false;
        }
    }
    service.release_terminal();
    result->milliseconds = std::chrono::duration<double, std::milli>(
        Clock::now() - start).count();
    for (const auto& request : requests) {
        result->generated.push_back(request->generated());
    }
    result->metrics = service.metrics();
    return true;
}

double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    const std::size_t middle = values.size() / 2;
    return values.size() % 2 == 0
        ? (values[middle - 1] + values[middle]) * 0.5 : values[middle];
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2 || argc > 3) {
        std::cerr << "usage: real_qwen15_moe_service_benchmark "
                     "<model.gguf> [iterations=2]\n";
        return 2;
    }
    const std::size_t iterations = argc == 3
        ? static_cast<std::size_t>(std::stoul(argv[2])) : 2;
    if (iterations == 0) return 2;
    std::string error;
    auto factory = cpp_llm_engine::RealQwen2MoeExecutorFactory::load(
        argv[1], &error);
    if (factory == nullptr) {
        std::cerr << "model load failed: " << error << '\n';
        return 1;
    }
    factory->set_layer_cache_capacity(factory->layer_count());
    std::vector<double> row_samples;
    std::vector<double> grouped_samples;
    bool outputs_match = true;
    bool lifecycle_match = true;
    std::size_t seed = 1;
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
        if (!run(*factory, first_mode, seed, &first, &error) ||
            !run(*factory, second_mode, seed + 4, &second, &error)) {
            std::cerr << "service benchmark failed: " << error << '\n';
            return 1;
        }
        seed += 8;
        const auto& row = grouped_first ? second : first;
        const auto& grouped = grouped_first ? first : second;
        row_samples.push_back(row.milliseconds);
        grouped_samples.push_back(grouped.milliseconds);
        outputs_match = outputs_match && row.generated == grouped.generated;
        lifecycle_match = lifecycle_match &&
            row.metrics.accepted_total == 4 && grouped.metrics.accepted_total == 4 &&
            row.metrics.completed_total == 4 && grouped.metrics.completed_total == 4 &&
            row.metrics.released_total == 4 && grouped.metrics.released_total == 4 &&
            row.model_batch_calls == grouped.model_batch_calls &&
            row.tensor_batch_requests == grouped.tensor_batch_requests;
        std::cout << "paired_iteration=" << iteration + 1
                  << " path_order="
                  << (grouped_first ? "grouped-first" : "row-wise-first")
                  << " row_wise_ms=" << row.milliseconds
                  << " grouped_ms=" << grouped.milliseconds << '\n';
    }
    const double row_median = median(row_samples);
    const double grouped_median = median(grouped_samples);
    std::cout << "requests: 4\nnew_tokens: 4\niterations: " << iterations
              << "\nrow_wise_median_ms: " << row_median
              << "\ngrouped_median_ms: " << grouped_median
              << "\ngrouped_to_row_wise_ratio: " << grouped_median / row_median
              << "\noutputs_match: " << (outputs_match ? "true" : "false")
              << "\nlifecycle_match: " << (lifecycle_match ? "true" : "false")
              << "\ncorrectness: "
              << (outputs_match && lifecycle_match ? "PASS" : "FAIL") << '\n';
    return outputs_match && lifecycle_match ? 0 : 1;
}
