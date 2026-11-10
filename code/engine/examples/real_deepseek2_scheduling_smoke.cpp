// Engine/Chapter08：验证 DeepSeek-V2-Lite 双请求进入 Continuous Batch。
#include "cpp_llm_engine/real_deepseek2_executor_factory.h"

#include "chapter06_generation/request.h"
#include "chapter06_generation/runtime.h"
#include "chapter06_generation/sampler.h"
#include "chapter08_scheduling/step_scheduler.h"

#include <iostream>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "usage: real_deepseek2_scheduling_smoke <model.gguf>\n";
        return 2;
    }
    std::string error;
    auto factory = cpp_llm_engine::RealDeepSeek2ExecutorFactory::load(
        argv[1], &error);
    if (factory == nullptr) {
        std::cerr << "DeepSeek-V2-Lite load failed: " << error << '\n';
        return 1;
    }
    const bool default_mode = factory->moe_batch_mode() ==
        cpp_llm_engine::RealDeepSeek2MoeBatchMode::Grouped;
    factory->set_moe_batch_mode(
        cpp_llm_engine::RealDeepSeek2MoeBatchMode::RowWise);
    const bool row_wise_mode = factory->moe_batch_mode() ==
        cpp_llm_engine::RealDeepSeek2MoeBatchMode::RowWise;
    factory->set_moe_batch_mode(
        cpp_llm_engine::RealDeepSeek2MoeBatchMode::Grouped);

    chapter06_generation::GenerationConfig config;
    config.max_new_tokens = 2;
    config.eos_token_id = factory->eos_token_id();
    chapter06_generation::Runtime runtime(config);
    chapter06_generation::GreedySampler sampler;
    chapter06_generation::Request request_a({0});
    chapter06_generation::Request request_b({0});
    auto executor_a = factory->create_executor();
    auto executor_b = factory->create_executor();
    const std::vector<chapter08_scheduling::ScheduledRequest> scheduled{
        {101, &request_a, &sampler, executor_a},
        {102, &request_b, &sampler, executor_b},
    };
    chapter08_scheduling::ContinuousBatchScheduler scheduler;
    const auto prefill = scheduler.run_step(
        runtime, scheduled, 2, 2, {}, nullptr, 0, nullptr, factory.get());
    const auto decode = scheduler.run_step(
        runtime, scheduled, 2, 2, {}, nullptr, 0, nullptr, factory.get());

    const bool prefill_match = prefill.prefill_requests == 2 &&
        prefill.generated_tokens == 2 && prefill.model_batch_calls == 0;
    const bool batch_match = decode.decode_requests == 2 &&
        decode.generated_tokens == 2 && decode.model_batch_calls == 1 &&
        decode.tensor_batch_requests == 2;
    const chapter06_generation::TokenSequence expected{185, 85770};
    const bool requests_match = request_a.finished() && request_b.finished() &&
        request_a.generated() == expected &&
        request_a.generated() == request_b.generated() &&
        request_a.history().size() == 3 && request_b.history().size() == 3;
    const bool contexts_match =
        executor_a->execution_state().committed_tokens == 2 &&
        executor_b->execution_state().committed_tokens == 2;

    std::cout << "prefill_requests: " << prefill.prefill_requests
              << "\nmodel_batch_calls: " << decode.model_batch_calls
              << "\ntensor_batch_requests: " << decode.tensor_batch_requests
              << "\nrequest_tokens:";
    for (const auto token : request_a.generated()) std::cout << ' ' << token;
    const bool passed =
        default_mode && row_wise_mode && prefill_match && batch_match &&
        requests_match && contexts_match;
    std::cout << "\ndeepseek2_scheduling: "
              << (passed ? "PASS" : "FAIL") << '\n';
    return passed ? 0 : 1;
}
