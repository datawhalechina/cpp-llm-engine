// Chapter 04：教学型 MoE Router、Top-K、专家 FFN 与共享专家测试。
#include "chapter04_attention/moe_feed_forward.h"
#include "cpp_llm_engine/testing/test.h"

#include <cmath>
#include <limits>

namespace {

chapter04_attention::Tensor identity(std::size_t size, float scale = 1.0F) {
    chapter04_attention::Tensor value(size, size);
    for (std::size_t i = 0; i < size; ++i) value.at(i, i) = scale;
    return value;
}

chapter04_attention::MoeExpertWeights make_expert(float down_scale) {
    return {identity(2), identity(2), identity(2, down_scale)};
}

chapter04_attention::MoeFeedForward make_moe(std::size_t top_k = 2) {
    chapter04_attention::Tensor router(2, 3);
    router.at(0, 0) = 1.0F;
    router.at(1, 1) = 1.0F;
    router.at(0, 2) = 1.0F;
    router.at(1, 2) = 1.0F;
    return chapter04_attention::MoeFeedForward(
        2, 2, top_k, std::move(router),
        {make_expert(1.0F), make_expert(3.0F), make_expert(2.0F)},
        {0.0F, 0.0F}, make_expert(1.0F));
}

}  // namespace

int main() {
    cpp_llm_engine::testing::TestSuite suite("chapter04_moe_feed_forward");
    suite.run("stable Top-K and shared expert match fixed golden", [&] {
        const auto moe = make_moe();
        CPP_LLM_REQUIRE(suite, moe.ready());
        CPP_LLM_CHECK(suite, moe.expert_count() == 3);
        CPP_LLM_CHECK(suite, moe.top_k() == 2);

        chapter04_attention::Tensor input(2, 2);
        input.at(0, 0) = 1.0F;
        input.at(1, 1) = 1.0F;
        const auto result = moe.forward(input);
        CPP_LLM_REQUIRE(suite, result.ready());
        CPP_LLM_REQUIRE(suite, result.routes.size() == 2);
        CPP_LLM_CHECK(suite, result.routes[0].expert_indices ==
                                       std::vector<std::size_t>({0, 2}));
        CPP_LLM_CHECK(suite, result.routes[1].expert_indices ==
                                       std::vector<std::size_t>({1, 2}));
        for (const auto& route : result.routes) {
            CPP_LLM_CHECK(suite, std::abs(route.weights[0] - 0.5F) < 1.0e-6F);
            CPP_LLM_CHECK(suite, std::abs(route.weights[1] - 0.5F) < 1.0e-6F);
        }
        CPP_LLM_CHECK(suite, std::abs(result.output.at(0, 0) - 1.4621172F) < 1.0e-5F);
        CPP_LLM_CHECK(suite, std::abs(result.output.at(0, 1)) < 1.0e-6F);
        CPP_LLM_CHECK(suite, std::abs(result.output.at(1, 0)) < 1.0e-6F);
        CPP_LLM_CHECK(suite, std::abs(result.output.at(1, 1) - 2.1931758F) < 1.0e-5F);
    });

    suite.run("invalid routing and non-finite input preserve prior result", [&] {
        const auto invalid_top_k = make_moe(4);
        CPP_LLM_CHECK(suite, !invalid_top_k.ready());

        auto moe = make_moe();
        chapter04_attention::Tensor input(1, 2);
        input.at(0, 0) = std::numeric_limits<float>::quiet_NaN();
        chapter04_attention::MoeForwardResult result;
        result.output = chapter04_attention::Tensor(1, 2, 7.0F);
        const auto status = moe.try_forward(input, &result);
        CPP_LLM_CHECK(suite, !status.is_ok());
        CPP_LLM_CHECK(suite, status.message() ==
                                       "MoE router produced a non-finite value");
        CPP_LLM_CHECK(suite, result.output.at(0, 0) == 7.0F);
    });
    return suite.finish();
}
