// Chapter 06：声明确定性和可配置的 logits -> token sampling policy。
// 契约：Sampler 与 model execution 分离，使 Request output decision 可以独立测试。
// 验证：Sampler 测试检查 greedy、temperature、top-k 和 top-p 行为。
#pragma once

#include "chapter06_generation/generation_types.h"

#include <vector>

namespace chapter06_generation {

class Sampler {
public:
    virtual ~Sampler() = default;
    virtual TokenId sample(const std::vector<float>& logits) const = 0;
};

class GreedySampler final : public Sampler {
public:
    TokenId sample(const std::vector<float>& logits) const override;
};

}  // namespace chapter06_generation
