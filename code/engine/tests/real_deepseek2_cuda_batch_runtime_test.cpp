// Engine/Chapter08: validate cross-request DeepSeek CUDA grouped MoE decode.
#include "cpp_llm_engine/real_deepseek2_cuda_executor_factory.h"

#include "chapter06_generation/sampler.h"

#include <cuda_runtime_api.h>

#include <iostream>
#include <string>
#include <vector>

#define CHECK(c) do { if (!(c)) { std::cerr << "check failed line " << __LINE__ << '\n'; return 1; } } while (false)

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "usage: real_deepseek2_cuda_batch_runtime_test <model.gguf>\n";
        return 2;
    }
    int devices = 0;
    CHECK(cudaGetDeviceCount(&devices) == cudaSuccess && devices > 0);

    std::string error;
    auto factory = cpp_llm_engine::RealDeepSeek2CudaExecutorFactory::load(
        argv[1], 8, &error);
    CHECK(factory != nullptr);
    auto first = factory->create_executor();
    auto second = factory->create_executor();
    CHECK(first != nullptr && second != nullptr);

    chapter06_generation::Logits first_prefill;
    chapter06_generation::Logits second_prefill;
    CHECK(first->prefill({11}, &first_prefill, &error));
    CHECK(second->prefill({11}, &second_prefill, &error));
    chapter06_generation::GreedySampler sampler;
    CHECK(sampler.sample(first_prefill) == 207);
    CHECK(sampler.sample(second_prefill) == 207);

    std::vector<chapter06_generation::Logits> logits;
    CHECK(!factory->decode_batch(
        {first, second}, {207, 207}, {1, 1}, {801, 801},
        &logits, &error));
    CHECK(first->execution_state().committed_tokens == 1);
    CHECK(second->execution_state().committed_tokens == 1);
    CHECK(factory->decode_batch(
        {first, second}, {207, 207}, {1, 1}, {801, 802},
        &logits, &error));
    CHECK(logits.size() == 2);
    CHECK(logits[0] == logits[1]);
    CHECK(first->execution_state().committed_tokens == 2);
    CHECK(second->execution_state().committed_tokens == 2);
    CHECK(factory->batch_workspace_count() == 1);
    CHECK(factory->batch_workspace_use_count(2) == 1);
    CHECK(factory->grouped_moe_call_count(2) == 26);
    CHECK(factory->active_expert_group_count(2) == 156);
    CHECK(factory->multi_row_expert_group_count(2) == 156);
    CHECK(factory->routed_row_count(2) == 312);

    std::cout << "batch_size: 2\n"
              << "workspace_count: 1\n"
              << "workspace_uses: 1\n"
              << "grouped_moe_calls: 26\n"
              << "multi_row_expert_groups: 156\n"
              << "deepseek2_cuda_batch_runtime: PASS\n";
    return 0;
}
