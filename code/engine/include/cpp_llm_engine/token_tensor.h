// Engine：token 序列轻量包装
// 职责：保存 tokenizer 输出，作为 Engine 到 Request 的只读输入对象。
// 契约：对象拥有 token vector，不负责模型权重或推理计算。
// 验证：code/engine/tests/engine_smoke_test.cpp。
#pragma once

#include "chapter06_generation/generation_types.h"

#include <cstddef>
#include <vector>

namespace cpp_llm_engine {

class TokenTensor {
public:
    explicit TokenTensor(std::vector<chapter06_generation::TokenId> values);

    const std::vector<chapter06_generation::TokenId>& values() const;
    std::size_t size() const;

private:
    std::vector<chapter06_generation::TokenId> values_;
};

}  // namespace cpp_llm_engine
