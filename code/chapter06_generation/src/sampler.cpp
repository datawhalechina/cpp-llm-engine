// Chapter 06：实现 logits -> token 的 sampling policy。
// 职责：把 sampling decision 与 Transformer、Runtime state mutation 分离。
// 验证：Sampler 测试检查确定性和配置化选择行为。
#include "chapter06_generation/sampler.h"

#include <algorithm>

namespace chapter06_generation {

TokenId GreedySampler::sample(const std::vector<float>& logits) const {
    if (logits.empty()) {
        return 0;
    }

    return static_cast<TokenId>(
        std::distance(logits.begin(), std::max_element(logits.begin(), logits.end())));
}

}  // namespace chapter06_generation
