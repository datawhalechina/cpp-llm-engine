// Engine：真实稠密模型多 token 生成 smoke。
// 职责：验证文本经过 tokenizer、完整 Decoder、greedy sampling 和 detokenize。
// 运行：real_tinyllama_generation_smoke <model.gguf> [max_new_tokens] [prompt] [golden_csv]。
#include "cpp_llm_engine/real_llama_engine.h"

#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

using GeneratedToken = typename decltype(cpp_llm_engine::GenerationResult::generated_tokens)::value_type;

std::vector<GeneratedToken> parse_token_csv(const std::string& csv) {
    std::vector<GeneratedToken> tokens;
    std::istringstream input(csv);
    std::string item;
    while (std::getline(input, item, ',')) {
        if (!item.empty()) tokens.push_back(static_cast<GeneratedToken>(std::stoll(item)));
    }
    return tokens;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2 || argc > 5) {
        std::cerr << "usage: real_tinyllama_generation_smoke <model.gguf> "
                     "[max_new_tokens] [prompt] [golden_csv]\n";
        return 2;
    }
    const std::size_t max_new_tokens = argc >= 3
        ? static_cast<std::size_t>(std::strtoul(argv[2], nullptr, 10)) : 4;
    const std::string prompt = argc >= 4 ? argv[3] : "Hello TinyLlama";
    std::string error;
    auto engine = cpp_llm_engine::RealLlamaEngine::load(argv[1], &error);
    if (engine == nullptr) {
        std::cerr << "engine load failed: " << error << '\n';
        return 1;
    }
    cpp_llm_engine::GenerationResult result;
    if (!engine->generate(prompt, max_new_tokens, &result, &error)) {
        std::cerr << "generation failed: " << error << '\n';
        return 1;
    }
    std::cout << "architecture: " << engine->architecture() << '\n'
              << "prompt: " << prompt << '\n'
              << "prompt_tokens: " << result.prompt_tokens.size() << '\n'
              << "generated_tokens:";
    for (const auto token : result.generated_tokens) std::cout << ' ' << token;
    std::cout << '\n'
              << "generated_text: " << result.output << '\n'
              << "finished: " << std::boolalpha << result.finished << '\n';
    bool golden_matches = true;
    bool context_isolation_matches = true;
    if (argc == 5) {
        golden_matches = result.generated_tokens == parse_token_csv(argv[4]);
        std::cout << "golden_tokens_match: " << golden_matches << '\n';
        cpp_llm_engine::GenerationResult repeated;
        context_isolation_matches = engine->generate(
            prompt, max_new_tokens, &repeated, &error) &&
            repeated.generated_tokens == result.generated_tokens &&
            repeated.output == result.output;
        std::cout << "context_isolation_match: " << context_isolation_matches << '\n';
    }
    std::cout << "real_generation: ready\n";
    return result.finished && !result.generated_tokens.empty() && golden_matches &&
        context_isolation_matches ? 0 : 1;
}
