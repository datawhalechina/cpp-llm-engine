// Engine/Chapter06: validate DeepSeek-V2-Lite CUDA through Generation Runtime.
#include "cpp_llm_engine/real_deepseek2_cuda_executor_factory.h"

#include "chapter06_generation/request.h"
#include "chapter06_generation/runtime.h"
#include "chapter06_generation/sampler.h"

#include <cuda_runtime_api.h>

#include <iostream>
#include <string>

#define CHECK(c) do { if (!(c)) { std::cerr << "check failed line " << __LINE__ << '\n'; return 1; } } while (false)

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "usage: real_deepseek2_cuda_runtime_test <model.gguf>\n";
        return 2;
    }
    int devices = 0;
    CHECK(cudaGetDeviceCount(&devices) == cudaSuccess && devices > 0);

    std::string error;
    auto factory = cpp_llm_engine::RealDeepSeek2CudaExecutorFactory::load(
        argv[1], 8, &error);
    CHECK(factory != nullptr);
    CHECK(factory->layer_count() == 27);
    CHECK(factory->vocab_size() > 207);
    CHECK(factory->context_capacity() == 8);

    chapter06_generation::GenerationConfig config;
    config.max_new_tokens = 1;
    config.eos_token_id = factory->eos_token_id();
    chapter06_generation::Runtime runtime(config);
    chapter06_generation::GreedySampler sampler;

    const auto generate = [&]() {
        auto executor = factory->create_executor();
        CHECK(executor != nullptr);
        chapter06_generation::Request request({11});
        CHECK(runtime.generate(request, sampler, *executor) == 1);
        CHECK(request.finished());
        CHECK(request.generated() == chapter06_generation::TokenSequence{207});
        const auto state = executor->execution_state();
        CHECK(state.available);
        CHECK(state.committed_tokens == 1);
        CHECK(state.kv_layer_count == 27);
        CHECK(state.stores_kv_values);
        return 0;
    };
    CHECK(generate() == 0);
    CHECK(generate() == 0);

    auto chunked = factory->create_executor();
    CHECK(chunked != nullptr && chunked->supports_chunked_prefill());
    chapter06_generation::Logits logits;
    CHECK(chunked->prefill_chunk({11}, true, false, nullptr, &error));
    CHECK(chunked->prefill_chunk({207}, false, true, &logits, &error));
    CHECK(!logits.empty());
    CHECK(chunked->execution_state().committed_tokens == 2);

    std::cout << "architecture: deepseek2\n"
              << "layers: " << factory->layer_count() << '\n'
              << "generated_token: 207\n"
              << "chunked_prefill_position: 2\n"
              << "context_isolation_match: true\n"
              << "deepseek2_cuda_runtime: PASS\n";
    return 0;
}
