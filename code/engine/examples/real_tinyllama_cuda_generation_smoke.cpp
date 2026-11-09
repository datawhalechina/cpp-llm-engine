// Engine CUDA：验证 Llama/Qwen3 文本经过 tokenizer、CUDA Runtime 和 detokenizer。
// 运行：real_tinyllama_cuda_generation_smoke <model.gguf> [budget] [prompt] [golden_csv]。
#include "cpp_llm_engine/real_llama_cuda_engine.h"

#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

using GeneratedToken = typename decltype(
    cpp_llm_engine::GenerationResult::generated_tokens)::value_type;

std::vector<GeneratedToken> parse_token_csv(const std::string& csv) {
    std::vector<GeneratedToken> tokens;
    std::istringstream input(csv);
    std::string item;
    while (std::getline(input, item, ',')) {
        if (!item.empty()) {
            tokens.push_back(static_cast<GeneratedToken>(std::stoll(item)));
        }
    }
    return tokens;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2 || argc > 5) {
        std::cerr << "usage: real_tinyllama_cuda_generation_smoke <model.gguf> "
                     "[max_new_tokens] [prompt] [golden_csv]\n";
        return 2;
    }
    const std::size_t max_new_tokens = argc >= 3
        ? static_cast<std::size_t>(std::strtoul(argv[2], nullptr, 10)) : 4;
    const std::string prompt = argc >= 4 ? argv[3] : "Hello TinyLlama";
    std::string error;
    auto engine = cpp_llm_engine::RealLlamaCudaEngine::load(
        argv[1], 512, &error);
    if (engine == nullptr) {
        std::cerr << "CUDA engine load failed: " << error << '\n';
        return 1;
    }
    cpp_llm_engine::GenerationResult result;
    if (!engine->generate(prompt, max_new_tokens, &result, &error)) {
        std::cerr << "CUDA generation failed: " << error << '\n';
        return 1;
    }
    bool golden_matches = true;
    bool context_isolation_matches = true;
    if (argc == 5) {
        golden_matches = result.generated_tokens == parse_token_csv(argv[4]);
        cpp_llm_engine::GenerationResult repeated;
        context_isolation_matches = engine->generate(
            prompt, max_new_tokens, &repeated, &error) &&
            repeated.generated_tokens == result.generated_tokens &&
            repeated.output == result.output;
    }
    std::cout << "architecture: " << engine->architecture() << '\n'
              << "prompt_tokens: " << result.prompt_tokens.size() << '\n'
              << "generated_tokens:";
    for (const auto token : result.generated_tokens) std::cout << ' ' << token;
    std::cout << "\ngenerated_text: " << result.output << '\n'
              << "golden_tokens_match: " << std::boolalpha << golden_matches << '\n'
              << "context_isolation_match: " << context_isolation_matches << '\n'
              << "cuda_text_generation: ready\n";
    return result.finished && !result.generated_tokens.empty() &&
        golden_matches && context_isolation_matches ? 0 : 1;
}
