// Engine：真实 Llama/Qwen3 CPU 生成入口。
// 职责：共享真实 tokenizer/Model，并为每次生成创建独立请求 Context。
// 验证：真实模型 CTest 检查 golden token 和重复生成的 context 隔离。
#pragma once

#include "cpp_llm_engine/engine.h"

#include <cstddef>
#include <memory>
#include <string>
#include <string_view>

namespace cpp_llm_engine {

class RealLlamaEngine final {
public:
    static std::unique_ptr<RealLlamaEngine> load(
        const std::string& model_path,
        std::string* error = nullptr);

    ~RealLlamaEngine();
    RealLlamaEngine(RealLlamaEngine&&) noexcept;
    RealLlamaEngine& operator=(RealLlamaEngine&&) noexcept;

    RealLlamaEngine(const RealLlamaEngine&) = delete;
    RealLlamaEngine& operator=(const RealLlamaEngine&) = delete;

    bool generate(std::string_view prompt,
                  std::size_t max_new_tokens,
                  GenerationResult* result,
                  std::string* error = nullptr);
    std::string architecture() const;

private:
    struct Impl;
    explicit RealLlamaEngine(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};

}  // namespace cpp_llm_engine
