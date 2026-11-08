// Chapter 06：实现 logits 到词元的 Greedy 采样。
// 职责：把词元选择与 Transformer、Runtime 状态变化分离。
// 验证：Sampler 测试检查确定性选择和空输入失败行为。
#include "chapter06_generation/sampler.h"

#include <algorithm>
#include <stdexcept>

namespace chapter06_generation {

TokenId GreedySampler::sample(const std::vector<float>& logits) const {
    if (logits.empty()) {
        throw std::invalid_argument("greedy sampling requires non-empty logits");
    }

    return static_cast<TokenId>(
        std::distance(logits.begin(), std::max_element(logits.begin(), logits.end())));
}

}  // namespace chapter06_generation
