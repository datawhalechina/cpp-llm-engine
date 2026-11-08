// Chapter 06：Greedy Sampler 的命名单元测试。
// 职责：验证 logits 最大值映射和空输入失败行为。
// 验证入口：带 unit、sampler 标签的 Chapter06 CTest。
#include "chapter06_generation/sampler.h"

#include "cpp_llm_engine/testing/test.h"

#include <stdexcept>

int main() {
    cpp_llm_engine::testing::TestSuite suite("chapter06_greedy_sampler");

    suite.run("selects index of maximum logit", [&] {
        chapter06_generation::GreedySampler sampler;
        CPP_LLM_CHECK(suite, sampler.sample({0.1f, 0.9f, 0.2f}) == 1);
    });

    suite.run("rejects empty logits instead of returning a valid token", [&] {
        chapter06_generation::GreedySampler sampler;
        bool rejected = false;
        try {
            (void)sampler.sample({});
        } catch (const std::invalid_argument&) {
            rejected = true;
        }
        CPP_LLM_CHECK(suite, rejected);
    });

    return suite.finish();
}
