// Engine：真实 Llama/Qwen3 CPU 生成入口实现。
// 职责：由 Chapter06 Runtime 驱动 prefill、greedy sampling、decode 和 EOS 停止。
// 契约：Engine 共享只读 Model，并为每次 generate 创建独立请求 Context。
#include "cpp_llm_engine/real_llama_engine.h"

#include "cpp_llm_engine/real_llama_executor_factory.h"
#include "cpp_llm_engine/real_tokenizer_backend.h"
#include "chapter06_generation/request.h"
#include "chapter06_generation/runtime.h"
#include "chapter06_generation/sampler.h"

#include <utility>

namespace cpp_llm_engine {

struct RealLlamaEngine::Impl {
    std::unique_ptr<RealTokenizerBackend> tokenizer;
    std::unique_ptr<RealLlamaExecutorFactory> executor_factory;
};

std::unique_ptr<RealLlamaEngine> RealLlamaEngine::load(
    const std::string& model_path,
    std::string* error) {
    auto tokenizer = RealTokenizerBackend::load(model_path, error);
    if (tokenizer == nullptr) return nullptr;
    auto executor_factory = RealLlamaExecutorFactory::load(model_path, error);
    if (executor_factory == nullptr) return nullptr;
    if (tokenizer->vocab_size() != executor_factory->vocab_size()) {
        if (error != nullptr) *error = "tokenizer and model vocabulary sizes differ";
        return nullptr;
    }
    auto impl = std::make_unique<Impl>();
    impl->tokenizer = std::move(tokenizer);
    impl->executor_factory = std::move(executor_factory);
    if (error != nullptr) error->clear();
    return std::unique_ptr<RealLlamaEngine>(new RealLlamaEngine(std::move(impl)));
}

RealLlamaEngine::RealLlamaEngine(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}

RealLlamaEngine::~RealLlamaEngine() = default;
RealLlamaEngine::RealLlamaEngine(RealLlamaEngine&&) noexcept = default;
RealLlamaEngine& RealLlamaEngine::operator=(RealLlamaEngine&&) noexcept = default;

bool RealLlamaEngine::generate(std::string_view prompt,
                               std::size_t max_new_tokens,
                               GenerationResult* result,
                               std::string* error) {
    if (result == nullptr || prompt.empty() || max_new_tokens == 0) {
        if (error != nullptr) *error = "generation requires prompt, token budget and result output";
        return false;
    }
    const auto prompt_tokens = impl_->tokenizer->encode(prompt);
    if (prompt_tokens.empty()) {
        if (error != nullptr) *error = "tokenizer produced an empty prompt";
        return false;
    }

    auto executor = impl_->executor_factory->create_executor();
    chapter06_generation::GenerationConfig config;
    config.max_new_tokens = max_new_tokens;
    config.eos_token_id = impl_->executor_factory->eos_token_id();
    chapter06_generation::Runtime runtime(config);
    chapter06_generation::GreedySampler sampler;
    chapter06_generation::Request request(prompt_tokens);
    runtime.generate(request, sampler, *executor);
    if (request.failed()) {
        if (error != nullptr) *error = request.failure_reason();
        return false;
    }

    *result = GenerationResult{
        request.prompt(),
        request.generated(),
        impl_->tokenizer->decode(request.generated()),
        request.finished(),
    };
    if (error != nullptr) error->clear();
    return true;
}

std::string RealLlamaEngine::architecture() const {
    return impl_->executor_factory->architecture();
}

}  // namespace cpp_llm_engine
