// Engine CUDA：实现真实文本到 CUDA Generation Runtime 的纵向入口。
// 职责：只编排 tokenizer、请求、采样和 Executor，不持有请求级 K/V。
// 验证：真实模型 smoke 对照 CPU golden token，并检查重复请求隔离。
#include "cpp_llm_engine/real_llama_cuda_engine.h"

#include "cpp_llm_engine/real_llama_cuda_executor_factory.h"
#include "cpp_llm_engine/real_tokenizer_backend.h"
#include "chapter06_generation/request.h"
#include "chapter06_generation/runtime.h"
#include "chapter06_generation/sampler.h"

#include <utility>

namespace cpp_llm_engine {

struct RealLlamaCudaEngine::Impl final {
    std::unique_ptr<RealTokenizerBackend> tokenizer;
    std::unique_ptr<RealLlamaCudaExecutorFactory> executor_factory;
};

std::unique_ptr<RealLlamaCudaEngine> RealLlamaCudaEngine::load(
    const std::string& model_path,
    std::size_t context_capacity,
    std::string* error) {
    auto tokenizer = RealTokenizerBackend::load(model_path, error);
    if (tokenizer == nullptr) return nullptr;
    auto executor_factory = RealLlamaCudaExecutorFactory::load(
        model_path, context_capacity, error);
    if (executor_factory == nullptr) return nullptr;
    if (tokenizer->vocab_size() != executor_factory->vocab_size()) {
        if (error != nullptr) *error = "tokenizer and CUDA model vocabulary sizes differ";
        return nullptr;
    }
    auto impl = std::make_unique<Impl>();
    impl->tokenizer = std::move(tokenizer);
    impl->executor_factory = std::move(executor_factory);
    if (error != nullptr) error->clear();
    return std::unique_ptr<RealLlamaCudaEngine>(
        new RealLlamaCudaEngine(std::move(impl)));
}

RealLlamaCudaEngine::RealLlamaCudaEngine(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}

RealLlamaCudaEngine::~RealLlamaCudaEngine() = default;
RealLlamaCudaEngine::RealLlamaCudaEngine(RealLlamaCudaEngine&&) noexcept = default;
RealLlamaCudaEngine& RealLlamaCudaEngine::operator=(
    RealLlamaCudaEngine&&) noexcept = default;

bool RealLlamaCudaEngine::generate(
    std::string_view prompt,
    std::size_t max_new_tokens,
    GenerationResult* result,
    std::string* error) {
    if (result == nullptr || prompt.empty() || max_new_tokens == 0) {
        if (error != nullptr) {
            *error = "CUDA generation requires prompt, token budget and output";
        }
        return false;
    }
    const auto prompt_tokens = impl_->tokenizer->encode(prompt);
    if (prompt_tokens.empty() ||
        prompt_tokens.size() + max_new_tokens >
            impl_->executor_factory->context_capacity()) {
        if (error != nullptr) {
            *error = prompt_tokens.empty()
                ? "tokenizer produced an empty prompt"
                : "prompt and generation budget exceed CUDA context capacity";
        }
        return false;
    }
    auto executor = impl_->executor_factory->create_executor();
    if (executor == nullptr) {
        if (error != nullptr) *error = "CUDA request context creation failed";
        return false;
    }
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

}  // namespace cpp_llm_engine
