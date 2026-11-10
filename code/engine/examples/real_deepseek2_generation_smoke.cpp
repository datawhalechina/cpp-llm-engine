// Engine/Chapter06：验证文本经 tokenizer、DeepSeek2 Runtime 和 detokenizer。
#include "cpp_llm_engine/real_deepseek2_executor_factory.h"
#include "cpp_llm_engine/real_tokenizer_backend.h"

#include "chapter06_generation/request.h"
#include "chapter06_generation/runtime.h"
#include "chapter06_generation/sampler.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>

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
    if (argc < 2 || argc > 3) {
        std::cerr << "usage: real_deepseek2_generation_smoke <model.gguf> [prompt]\n";
        return 2;
    }
    const std::string prompt = argc == 3 ? argv[2] : "Hello";
    std::string error;
    auto tokenizer = cpp_llm_engine::RealTokenizerBackend::load(argv[1], &error);
    auto factory = cpp_llm_engine::RealDeepSeek2ExecutorFactory::load(
        argv[1], &error);
    if (tokenizer == nullptr || factory == nullptr) {
        std::cerr << "DeepSeek-V2-Lite load failed: " << error << '\n';
        return 1;
    }
    const auto prompt_tokens = tokenizer->encode(prompt);
    if (prompt_tokens.empty()) {
        std::cerr << "tokenizer produced an empty prompt\n";
        return 1;
    }

    chapter06_generation::GenerationConfig config;
    config.max_new_tokens = 2;
    config.eos_token_id = factory->eos_token_id();
    chapter06_generation::Runtime runtime(config);
    chapter06_generation::GreedySampler sampler;
    chapter06_generation::Request request(prompt_tokens);
    auto executor = factory->create_executor();
    if (runtime.generate(request, sampler, *executor) != 2 ||
        !request.finished()) {
        std::cerr << "DeepSeek-V2-Lite text generation failed\n";
        return 1;
    }
    const chapter06_generation::TokenSequence golden{11, 304};
    if (prompt == "Hello" && request.generated() != golden) {
        std::cerr << "DeepSeek-V2-Lite generated token golden changed\n";
        return 1;
    }
    const auto state = executor->execution_state();
    const auto cache = factory->layer_cache_stats();
    if (!state.available || state.kv_layer_count != factory->layer_count() ||
        state.committed_tokens != prompt_tokens.size() + 1 ||
        cache.capacity != 0 || cache.resident_layers != 0 ||
        cache.resident_bytes != 0 || cache.hits != 0 || cache.misses != 54) {
        std::cerr << "DeepSeek-V2-Lite execution state is inconsistent\n";
        return 1;
    }

    auto batch_a = factory->create_executor();
    auto batch_b = factory->create_executor();
    chapter06_generation::Logits ignored;
    if (!batch_a->prefill(prompt_tokens, &ignored, &error) ||
        !batch_b->prefill(prompt_tokens, &ignored, &error)) {
        std::cerr << "DeepSeek-V2-Lite batch prefill failed: " << error << '\n';
        return 1;
    }
    std::vector<chapter06_generation::Logits> batch_logits;
    if (!factory->decode_batch(
            {batch_a, batch_b}, {11, 11},
            {prompt_tokens.size(), prompt_tokens.size()}, {101, 102},
            &batch_logits, &error) || batch_logits.size() != 2) {
        std::cerr << "DeepSeek-V2-Lite batch decode failed: " << error << '\n';
        return 1;
    }
    const float batch_error = maximum_error(batch_logits[0], batch_logits[1]);
    if (batch_error > 1.0e-5F ||
        batch_a->execution_state().committed_tokens != prompt_tokens.size() + 1 ||
        batch_b->execution_state().committed_tokens != prompt_tokens.size() + 1) {
        std::cerr << "DeepSeek-V2-Lite batch results diverged\n";
        return 1;
    }
    std::vector<chapter06_generation::Logits> sentinel(1);
    if (factory->decode_batch(
            {batch_a, batch_b}, {11, 11},
            {prompt_tokens.size() + 1, prompt_tokens.size() + 1}, {101, 101},
            &sentinel, &error) || sentinel.size() != 1 ||
        batch_a->execution_state().committed_tokens != prompt_tokens.size() + 1 ||
        batch_b->execution_state().committed_tokens != prompt_tokens.size() + 1) {
        std::cerr << "DeepSeek-V2-Lite invalid batch changed request state\n";
        return 1;
    }
    std::cout << "architecture: deepseek2\n"
              << "layers: " << factory->layer_count() << '\n'
              << "prompt_tokens: " << prompt_tokens.size() << '\n'
              << "layer_cache_misses: " << cache.misses << '\n'
              << "batch_logits_max_error: " << batch_error << '\n'
              << "generated_tokens:";
    for (const auto token : request.generated()) std::cout << ' ' << token;
    std::cout << "\ngenerated_text: "
              << tokenizer->decode(request.generated()) << '\n'
              << "golden_tokens_match: true\n"
              << "deepseek2_text_generation: PASS\n";
    return 0;
}
