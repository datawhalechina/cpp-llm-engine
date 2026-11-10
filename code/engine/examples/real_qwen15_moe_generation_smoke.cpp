// Engine：验证文本经 Qwen tokenizer、MoE Runtime 和 detokenizer 的完整链路。
#include "cpp_llm_engine/real_qwen2_moe_executor_factory.h"
#include "cpp_llm_engine/real_tokenizer_backend.h"

#include "chapter06_generation/request.h"
#include "chapter06_generation/runtime.h"
#include "chapter06_generation/sampler.h"

#include <iostream>
#include <string>

int main(int argc, char** argv) {
    if (argc < 2 || argc > 3) {
        std::cerr << "usage: real_qwen15_moe_generation_smoke <model.gguf> [prompt]\n";
        return 2;
    }
    const std::string prompt = argc == 3 ? argv[2] : "Hello";
    std::string error;
    auto tokenizer = cpp_llm_engine::RealTokenizerBackend::load(argv[1], &error);
    auto factory = cpp_llm_engine::RealQwen2MoeExecutorFactory::load(
        argv[1], &error);
    if (tokenizer == nullptr || factory == nullptr) {
        std::cerr << "Qwen1.5-MoE load failed: " << error << '\n';
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
        std::cerr << "Qwen1.5-MoE text generation failed\n";
        return 1;
    }
    std::cout << "prompt_tokens: " << prompt_tokens.size() << '\n'
              << "generated_tokens:";
    for (const auto token : request.generated()) std::cout << ' ' << token;
    std::cout << "\ngenerated_text: "
              << tokenizer->decode(request.generated()) << '\n'
              << "qwen15_moe_text_generation: PASS\n";
    return 0;
}
