// Chapter 06：Greedy Sampler 的命名单元测试。
// 职责：验证 logits 最大值稳定映射到 token id。
// 验证入口：带 unit、sampler 标签的 Chapter06 CTest。
#include "chapter06_generation/sampler.h"

#include "cpp_llm_engine/testing/test.h"

int main() {
    cpp_llm_engine::testing::TestSuite suite("chapter06_greedy_sampler");

    suite.run("selects index of maximum logit", [&] {
        chapter06_generation::GreedySampler sampler;
        CPP_LLM_CHECK(suite, sampler.sample({0.1f, 0.9f, 0.2f}) == 1);
    });

    return suite.finish();
}
