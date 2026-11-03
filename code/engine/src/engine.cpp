// Engine：文本生成入口实现
// 职责：执行 tokenizer -> Request -> Runtime -> token decode 的教学主线。
// 约束：当前使用 Chapter 06 教学 Transformer，不代表真实 GGUF 模型推理。
// 验证：code/engine/tests/engine_smoke_test.cpp 和 code/engine/scripts/run.sh。
#include "cpp_llm_engine/engine.h"

#include "chapter06_generation/embedding.h"
#include "chapter06_generation/position_encoding.h"
#include "chapter06_generation/request.h"
#include "chapter06_generation/runtime.h"
#include "chapter06_generation/sampler.h"
#include "chapter06_generation/transformer.h"

#include <stdexcept>

namespace cpp_llm_engine {
namespace {

class ToyTokenizerBackend final : public TokenizerBackend {
public:
    explicit ToyTokenizerBackend(std::size_t vocab_size) : tokenizer_(vocab_size) {}

    std::size_t vocab_size() const override { return tokenizer_.vocab_size(); }
    chapter01::TokenSequence encode(std::string_view text) const override {
        return tokenizer_.encode(text);
    }
    std::string decode(const chapter01::TokenSequence& tokens) const override {
        return tokenizer_.decode(tokens);
    }

private:
    chapter01::Tokenizer tokenizer_;
};

}  // namespace

TokenTensor::TokenTensor(std::vector<chapter06_generation::TokenId> values)
    : values_(std::move(values)) {}

const std::vector<chapter06_generation::TokenId>& TokenTensor::values() const {
    return values_;
}

std::size_t TokenTensor::size() const {
    return values_.size();
}

Engine::Engine(ModelConfig config, std::unique_ptr<TokenizerBackend> tokenizer)
    : config_(config),
      tokenizer_(tokenizer != nullptr
                     ? std::move(tokenizer)
                     : std::make_unique<ToyTokenizerBackend>(config.vocab_size)) {
    if (!config_.valid()) {
        throw std::invalid_argument("invalid model configuration");
    }
}

GenerationResult Engine::generate(std::string_view prompt) const {
    const TokenTensor input(tokenizer_->encode(prompt));

    chapter06_generation::GenerationConfig config;
    config.max_new_tokens = config_.max_new_tokens;
    config.use_kv_cache = true;

    chapter06_generation::Embedding embedding(config_.vocab_size);
    chapter06_generation::PositionEncoding position_encoding(config_.max_position);
    chapter06_generation::Transformer transformer(config_.layer_count, embedding, position_encoding);
    chapter06_generation::Runtime runtime(config, transformer);
    chapter06_generation::GreedySampler sampler;
    chapter06_generation::Request request(input.values());

    runtime.generate(request, sampler);

    return GenerationResult{
        request.prompt(),
        request.generated(),
        tokenizer_->decode(request.generated()),
        request.finished(),
    };
}

}  // namespace cpp_llm_engine
