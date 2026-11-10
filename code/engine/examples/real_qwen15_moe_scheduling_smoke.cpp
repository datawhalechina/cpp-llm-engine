// Engine/Chapter 08：验证 Qwen1.5-MoE 双请求进入 Continuous Batch。
// 通过条件：Prefill 后的 Decode 只调用一次 batch runner，并推进两个独立 Context。
#include "cpp_llm_engine/real_qwen2_moe_executor_factory.h"

#include "chapter06_generation/request.h"
#include "chapter06_generation/runtime.h"
#include "chapter06_generation/sampler.h"
#include "chapter08_scheduling/step_scheduler.h"

#include <iostream>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "usage: real_qwen15_moe_scheduling_smoke <model.gguf>\n";
        return 2;
    }
    std::string error;
    auto factory = cpp_llm_engine::RealQwen2MoeExecutorFactory::load(
        argv[1], &error);
    if (factory == nullptr) {
        std::cerr << "Qwen1.5-MoE load failed: " << error << '\n';
        return 1;
    }
    factory->set_layer_cache_capacity(1);

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

    const chapter06_generation::TokenSequence expected{16, 198};
    const bool prefill_match = prefill.prefill_requests == 2 &&
        prefill.generated_tokens == 2 && prefill.model_batch_calls == 0;
    const bool batch_match = decode.decode_requests == 2 &&
        decode.generated_tokens == 2 && decode.model_batch_calls == 1 &&
        decode.tensor_batch_requests == 2;
    const bool requests_match = request_a.finished() && request_b.finished() &&
        request_a.generated() == expected && request_b.generated() == expected &&
        request_a.history().size() == 3 && request_b.history().size() == 3;
    const bool contexts_match =
        executor_a->execution_state().committed_tokens == 2 &&
        executor_b->execution_state().committed_tokens == 2;

    std::cout << "prefill_requests: " << prefill.prefill_requests
              << "\nmodel_batch_calls: " << decode.model_batch_calls
              << "\ntensor_batch_requests: " << decode.tensor_batch_requests
              << "\nrequest_a_tokens:";
    for (const auto token : request_a.generated()) std::cout << ' ' << token;
    std::cout << "\nrequest_b_tokens:";
    for (const auto token : request_b.generated()) std::cout << ' ' << token;
    std::cout << "\nqwen15_moe_scheduling: "
              << (prefill_match && batch_match && requests_match && contexts_match
                      ? "PASS" : "FAIL")
              << '\n';
    return prefill_match && batch_match && requests_match && contexts_match
        ? 0 : 1;
}
