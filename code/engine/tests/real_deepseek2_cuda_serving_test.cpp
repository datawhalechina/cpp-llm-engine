// Engine/Chapter10: validate DeepSeek-V2-Lite CUDA service lifecycle.
#include "cpp_llm_engine/real_deepseek2_cuda_executor_factory.h"

#include "chapter06_generation/request.h"
#include "chapter06_generation/runtime.h"
#include "chapter06_generation/sampler.h"
#include "chapter10_serving/service_core.h"

#include <cuda_runtime_api.h>

#include <iostream>
#include <string>
#include <vector>

#define CHECK(c) do { if (!(c)) { std::cerr << "check failed line " << __LINE__ << '\n'; return 1; } } while (false)

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "usage: real_deepseek2_cuda_serving_test <model.gguf>\n";
        return 2;
    }
    int devices = 0;
    CHECK(cudaGetDeviceCount(&devices) == cudaSuccess && devices > 0);

    std::string error;
    auto factory = cpp_llm_engine::RealDeepSeek2CudaExecutorFactory::load(
        argv[1], 8, &error);
    CHECK(factory != nullptr);

    chapter06_generation::GenerationConfig config;
    config.max_new_tokens = 2;
    config.eos_token_id = factory->eos_token_id();
    chapter06_generation::Runtime runtime(config);
    chapter06_generation::GreedySampler sampler;
    chapter10_serving::ServiceCore service(3, 3, {}, {}, {}, factory.get());
    chapter06_generation::Request completed_a({11});
    chapter06_generation::Request completed_b({11});
    chapter06_generation::Request cancelled({11});
    CHECK(service.admit(
        501, completed_a, sampler, factory->create_executor()).accepted());
    CHECK(service.admit(
        502, completed_b, sampler, factory->create_executor()).accepted());
    CHECK(service.admit(
        503, cancelled, sampler, factory->create_executor()).accepted());
    CHECK(service.cancel(503));

    std::vector<chapter06_generation::BatchTokenResult> streamed;
    const auto stream_token =
        [&streamed](std::size_t request_id,
                    chapter06_generation::TokenId token,
                    std::size_t step) {
            streamed.push_back({request_id, token, step});
        };
    const auto prefill = service.dispatch_step(runtime, 3, stream_token);
    CHECK(prefill.generated_tokens == 2);
    CHECK(prefill.model_batch_calls == 0);
    CHECK(!completed_a.terminal() && !completed_b.terminal());
    CHECK(completed_a.generated() == chapter06_generation::TokenSequence{207});
    CHECK(completed_b.generated() == completed_a.generated());

    const auto decode = service.dispatch_step(runtime, 2, stream_token);
    CHECK(decode.generated_tokens == 2);
    CHECK(decode.model_batch_calls == 1);
    CHECK(decode.tensor_batch_requests == 2);
    CHECK(completed_a.finished() && completed_b.finished());
    CHECK(completed_a.generated().size() == 2);
    CHECK(completed_b.generated() == completed_a.generated());
    CHECK(cancelled.cancelled());
    CHECK(streamed.size() == 4);
    CHECK(streamed[0].batch_index == 501);
    CHECK(streamed[0].token == 207);
    CHECK(streamed[0].step == 0);
    CHECK(streamed[1].batch_index == 502);
    CHECK(streamed[1].token == 207);
    CHECK(streamed[1].step == 0);
    CHECK(streamed[2].batch_index == 501);
    CHECK(streamed[3].batch_index == 502);
    CHECK(streamed[2].step == 1 && streamed[3].step == 1);
    CHECK(streamed[2].token == streamed[3].token);
    CHECK(factory->batch_workspace_count() == 1);
    CHECK(factory->batch_workspace_use_count(2) == 1);
    CHECK(factory->grouped_moe_call_count(2) == 26);
    CHECK(service.release_terminal() == 3);
    CHECK(service.inflight_size() == 0);

    const auto& metrics = service.metrics();
    CHECK(metrics.accepted_total == 3);
    CHECK(metrics.completed_total == 2);
    CHECK(metrics.cancelled_total == 1);
    CHECK(metrics.released_total == 3);
    CHECK(metrics.generated_tokens_total == 4);

    std::cout << "generated_tokens:";
    for (const auto token : completed_a.generated()) std::cout << ' ' << token;
    std::cout << "\nmodel_batch_calls: " << decode.model_batch_calls
              << "\ntensor_batch_requests: " << decode.tensor_batch_requests
              << "\ngrouped_moe_calls: "
              << factory->grouped_moe_call_count(2)
              << "\nstream_event_count: " << streamed.size() << '\n'
              << "cancelled_total: 1\n"
              << "released_total: 3\n"
              << "deepseek2_cuda_serving: PASS\n";
    return 0;
}
