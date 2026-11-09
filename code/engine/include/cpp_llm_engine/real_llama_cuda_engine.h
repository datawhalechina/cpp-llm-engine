// Engine CUDA：声明真实 Llama/Qwen3 文本生成入口。
// 职责：连接真实 tokenizer、CUDA Executor、Chapter06 Runtime 和 detokenizer。
// 契约：每次 generate 创建独立请求 Context，Engine 只共享 tokenizer 与 Model。
#pragma once

#include "cpp_llm_engine/engine.h"

#include <cstddef>
#include <memory>
#include <string>
#include <string_view>

namespace cpp_llm_engine {

class RealLlamaCudaEngine final {
public:
    static std::unique_ptr<RealLlamaCudaEngine> load(
        const std::string& model_path,
        std::size_t context_capacity = 512,
        std::string* error = nullptr);

    ~RealLlamaCudaEngine();
    RealLlamaCudaEngine(RealLlamaCudaEngine&&) noexcept;
    RealLlamaCudaEngine& operator=(RealLlamaCudaEngine&&) noexcept;

    RealLlamaCudaEngine(const RealLlamaCudaEngine&) = delete;
    RealLlamaCudaEngine& operator=(const RealLlamaCudaEngine&) = delete;

    bool generate(
        std::string_view prompt,
        std::size_t max_new_tokens,
        GenerationResult* result,
        std::string* error = nullptr);
    std::string architecture() const;

private:
    struct Impl;
    explicit RealLlamaCudaEngine(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};

}  // namespace cpp_llm_engine
