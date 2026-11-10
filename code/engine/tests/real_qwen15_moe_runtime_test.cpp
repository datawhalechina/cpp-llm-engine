// Engine/Chapter 06：验证 Qwen1.5-MoE Executor 接入 Generation Runtime。
#include "cpp_llm_engine/real_qwen2_moe_executor_factory.h"

#include "chapter06_generation/request.h"
#include "chapter06_generation/runtime.h"
#include "chapter06_generation/sampler.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

#define CHECK(c) do { if (!(c)) { std::cerr << "check failed line " << __LINE__ << '\n'; return 1; } } while (false)

namespace {

float maximum_error(const chapter06_generation::Logits& left,
                    const chapter06_generation::Logits& right) {
    if (left.size() != right.size()) return INFINITY;
    float error = 0.0F;
    for (std::size_t index = 0; index < left.size(); ++index) {
        error = std::max(error, std::fabs(left[index] - right[index]));
    }
    return error;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "usage: real_qwen15_moe_runtime_test <model.gguf>\n";
        return 2;
    }
    std::string error;
    auto factory = cpp_llm_engine::RealQwen2MoeExecutorFactory::load(
        argv[1], &error);
    CHECK(factory != nullptr);
    CHECK(factory->layer_count() == 24);
    CHECK(factory->moe_batch_mode() ==
          cpp_llm_engine::RealQwen2MoeBatchMode::Grouped);
    factory->set_moe_batch_mode(
        cpp_llm_engine::RealQwen2MoeBatchMode::RowWise);
    CHECK(factory->moe_batch_mode() ==
          cpp_llm_engine::RealQwen2MoeBatchMode::RowWise);
    factory->set_moe_batch_mode(
        cpp_llm_engine::RealQwen2MoeBatchMode::Grouped);
    factory->set_layer_cache_capacity(1);

    chapter06_generation::GenerationConfig config;
    config.max_new_tokens = 2;
    config.eos_token_id = factory->eos_token_id();
    chapter06_generation::Runtime runtime(config);
    chapter06_generation::GreedySampler sampler;
    chapter06_generation::Request request({0});
    auto executor = factory->create_executor();
    CHECK(executor != nullptr);
    CHECK(runtime.generate(request, sampler, *executor) == 2);
    CHECK(request.finished());
    CHECK(request.generated().size() == 2);
    const auto state = executor->execution_state();
    CHECK(state.available);
    CHECK(state.committed_tokens == 2);
    CHECK(state.kv_layer_count == 24);
    CHECK(state.stores_kv_values);
    const auto cache = factory->layer_cache_stats();
    CHECK(cache.capacity == 1);
    CHECK(cache.resident_layers == 1);
    CHECK(cache.resident_bytes > 0);
    CHECK(cache.hits == 0);
    CHECK(cache.misses == 48);

    factory->set_layer_cache_capacity(factory->layer_count());
    auto batched_a = factory->create_executor();
    auto batched_b = factory->create_executor();
    chapter06_generation::Logits ignored;
    for (const auto& item : {batched_a, batched_b}) {
        CHECK(item->prefill({0}, &ignored, &error));
    }
    std::vector<chapter06_generation::Logits> batch_logits;
    CHECK(factory->decode_batch(
        {batched_a, batched_b}, {11, 11}, {1, 1}, {101, 102},
        &batch_logits, &error));
    CHECK(batch_logits.size() == 2);
    const float batch_error = maximum_error(batch_logits[0], batch_logits[1]);
    CHECK(batch_error <= 1.0e-5F);
    CHECK(sampler.sample(batch_logits[0]) == sampler.sample(batch_logits[1]));
    CHECK(batched_a->execution_state().committed_tokens == 2);
    CHECK(batched_b->execution_state().committed_tokens == 2);
    std::vector<chapter06_generation::Logits> sentinel(1);
    CHECK(!factory->decode_batch(
        {batched_a, batched_b}, {11, 11}, {2, 3}, {101, 102},
        &sentinel, &error));
    CHECK(batched_a->execution_state().committed_tokens == 2);
    CHECK(batched_b->execution_state().committed_tokens == 2);
    CHECK(sentinel.size() == 1);
    std::cout << "generated_tokens:";
    for (const auto token : request.generated()) std::cout << ' ' << token;
    std::cout << "\nlayer_cache_hits: " << cache.hits
              << "\nlayer_cache_misses: " << cache.misses
              << "\nbatch_logits_max_error: " << batch_error
              << "\nqwen15_moe_runtime: PASS\n";
    return 0;
}
