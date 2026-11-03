// Engine：教程级生成服务入口
// 职责：连接 tokenizer、Request 和 Chapter 06 Runtime，返回生成结果。
// 契约：Engine::generate 接收文本 prompt，输出 token 和解码文本。
// 验证：code/engine/tests/engine_smoke_test.cpp。
#pragma once

#include "chapter06_generation/generation_types.h"
#include "cpp_llm_engine/token_tensor.h"
#include "cpp_llm_engine/tokenizer_backend.h"
#include "model_config.h"
#include "tokenizer.h"

#include <cstddef>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace cpp_llm_engine {

using ModelConfig = chapter01::ModelConfig;

struct GenerationResult {
    std::vector<chapter06_generation::TokenId> prompt_tokens;
    std::vector<chapter06_generation::TokenId> generated_tokens;
    std::string output;
    bool finished = false;
};

class Engine {
public:
    explicit Engine(ModelConfig config = {},
                    std::unique_ptr<TokenizerBackend> tokenizer = nullptr);

    GenerationResult generate(std::string_view prompt) const;

private:
    ModelConfig config_;
    std::unique_ptr<TokenizerBackend> tokenizer_;
};

}  // namespace cpp_llm_engine
