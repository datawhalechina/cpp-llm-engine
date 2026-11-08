// Chapter 06：声明 logits 到词元的采样接口和 Greedy 基线。
// 契约：Sampler 与模型执行分离；空 logits 不能产生合法词元。
// 验证：Sampler 测试检查最大值选择和空输入失败行为。
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
