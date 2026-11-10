// Engine/Chapter10：验证 DeepSeek-V2-Lite 动态准入、组批与终态释放。
#include "cpp_llm_engine/real_deepseek2_executor_factory.h"

#include "chapter06_generation/request.h"
#include "chapter06_generation/runtime.h"
#include "chapter06_generation/sampler.h"
#include "chapter10_serving/service_core.h"

#include <algorithm>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "usage: real_deepseek2_service_smoke <model.gguf>\n";
        return 2;
    }
    std::string error;
    auto factory = cpp_llm_engine::RealDeepSeek2ExecutorFactory::load(
        argv[1], &error);
    if (factory == nullptr) {
        std::cerr << "DeepSeek-V2-Lite load failed: " << error << '\n';
        return 1;
    }

    chapter06_generation::GenerationConfig config;
    config.max_new_tokens = 2;
    config.eos_token_id = factory->eos_token_id();
    chapter06_generation::Runtime runtime(config);
    chapter06_generation::GreedySampler sampler;
    chapter10_serving::ServiceCore service(4, 4, {}, {}, {}, factory.get());
    std::vector<std::unique_ptr<chapter06_generation::Request>> requests;
    for (std::size_t index = 0; index < 4; ++index) {
        requests.push_back(std::make_unique<chapter06_generation::Request>(
            chapter06_generation::TokenSequence{0}));
    }
    const auto admit = [&](std::size_t index) {
        return service.admit(
            100 + index, *requests[index], sampler,
            factory->create_executor()).accepted();
    };
    if (!admit(0) || !admit(1)) {
        std::cerr << "initial DeepSeek service admission failed\n";
        return 1;
    }
    const auto initial_prefill = service.dispatch_step(runtime, 4);
    if (initial_prefill.generated_tokens != 2 ||
        initial_prefill.model_batch_calls != 0 || !admit(2) || !admit(3)) {
        std::cerr << "DeepSeek service Prefill or dynamic admission failed\n";
        return 1;
    }

    std::size_t model_batch_calls = 0;
    std::size_t tensor_batch_requests = 0;
    std::size_t released = 0;
    std::size_t steps = 0;
    while (std::any_of(
        requests.begin(), requests.end(),
        [](const auto& request) { return !request->terminal(); })) {
        const auto stats = service.dispatch_step(runtime, 4);
        model_batch_calls += stats.model_batch_calls;
        tensor_batch_requests += stats.tensor_batch_requests;
        released += service.release_terminal();
        if (stats.generated_tokens == 0 || ++steps > 4) {
            std::cerr << "DeepSeek service dispatch made no progress\n";
            return 1;
        }
    }
    released += service.release_terminal();

    const chapter06_generation::TokenSequence expected{185, 85770};
    const bool outputs_match = std::all_of(
        requests.begin(), requests.end(), [&expected](const auto& request) {
            return request->finished() && request->generated() == expected &&
                request->history().empty();
        });
    const auto& metrics = service.metrics();
    const bool lifecycle_match = metrics.accepted_total == 4 &&
        metrics.completed_total == 4 && metrics.released_total == 4 &&
        released == 4 && service.inflight_size() == 0;
    const bool batch_match =
        model_batch_calls == 2 && tensor_batch_requests == 4;
    const bool passed = outputs_match && lifecycle_match && batch_match;
    std::cout << "accepted: " << metrics.accepted_total
              << "\ncompleted: " << metrics.completed_total
              << "\nreleased: " << metrics.released_total
              << "\nmodel_batch_calls: " << model_batch_calls
              << "\ntensor_batch_requests: " << tensor_batch_requests
              << "\ndeepseek2_service: " << (passed ? "PASS" : "FAIL")
              << '\n';
    return passed ? 0 : 1;
}
