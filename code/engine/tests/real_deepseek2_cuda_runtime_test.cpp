// Engine/Chapter06: validate DeepSeek-V2-Lite CUDA through Generation Runtime.
#include "cpp_llm_engine/real_deepseek2_cuda_executor_factory.h"

#include "chapter06_generation/request.h"
#include "chapter06_generation/runtime.h"
#include "chapter06_generation/sampler.h"
#include "chapter07_kv_cache/block_table.h"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
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

    CHECK(factory->configure_paged_mla_cache(64, 2, &error));
    const auto paged_resource = factory->paged_mla_resource();
    CHECK(paged_resource != nullptr);
    auto paged = factory->create_executor();
    CHECK(paged != nullptr);
    CHECK(factory->bind_request(paged, 8101, &error));
    chapter06_generation::Logits paged_logits;
    CHECK(paged->prefill_chunk({11}, true, false, nullptr, &error));
    CHECK(paged_resource->token_count(8101) == 1);
    CHECK(paged->prefill_chunk({207}, false, true, &paged_logits, &error));
    CHECK(paged_resource->token_count(8101) == 2);
    CHECK(paged_logits.size() == logits.size());
    float paged_max_error = 0.0F;
    for (std::size_t index = 0; index < logits.size(); ++index) {
        paged_max_error = std::max(
            paged_max_error, std::fabs(logits[index] - paged_logits[index]));
    }
    CHECK(paged_max_error <= 1.0e-4F);
    CHECK(std::max_element(logits.begin(), logits.end()) - logits.begin() ==
        std::max_element(paged_logits.begin(), paged_logits.end()) -
            paged_logits.begin());

    auto paged_a = factory->create_executor();
    auto paged_b = factory->create_executor();
    CHECK(paged_a != nullptr && paged_b != nullptr);
    CHECK(factory->bind_request(paged_a, 8102, &error));
    CHECK(factory->bind_request(paged_b, 8103, &error));
    CHECK(paged_a->prefill_chunk({11}, true, true, &paged_logits, &error));
    CHECK(paged_b->prefill_chunk({11}, true, true, &paged_logits, &error));
    std::vector<chapter06_generation::Logits> batch_logits;
    CHECK(factory->decode_batch(
        {paged_a, paged_b},
        {207, 207},
        {1, 1},
        {8102, 8103},
        &batch_logits,
        &error));
    CHECK(batch_logits.size() == 2);
    CHECK(paged_resource->token_count(8102) == 2);
    CHECK(paged_resource->token_count(8103) == 2);

    factory->release_request(8101);
    factory->release_request(8102);
    factory->release_request(8103);
    CHECK(paged_resource->used_blocks() == 0);

    std::cout << "architecture: deepseek2\n"
              << "layers: " << factory->layer_count() << '\n'
              << "generated_token: 207\n"
              << "chunked_prefill_position: 2\n"
              << "paged_mla_max_error: " << paged_max_error << '\n'
              << "paged_mla_batch: true\n"
              << "paged_mla_terminal_release: true\n"
              << "context_isolation_match: true\n"
              << "deepseek2_cuda_runtime: PASS\n";
    return 0;
}
