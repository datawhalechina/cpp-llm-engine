// Model：验证真实量化 MoE 适配器的路由与专家执行。
// 职责：用独立可解析的 Q5_1 常量块核对 Top-K、SwiGLU 和共享专家。
// 通过条件：选中目标专家，输出符合解析式，失败保持调用方结果。
#include "cpp_llm_engine/model/real_moe_feed_forward.h"
#include "cpp_llm_engine/testing/test.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

namespace {

std::vector<std::byte> q5_1_ones(std::size_t matrix_count) {
    constexpr std::size_t kRows = 32;
    constexpr std::size_t kBlockBytes = 24;
    std::vector<std::byte> bytes(matrix_count * kRows * kBlockBytes,
                                 std::byte{0});
    for (std::size_t block = 0; block < matrix_count * kRows; ++block) {
        auto* raw = reinterpret_cast<std::uint8_t*>(
            bytes.data() + block * kBlockBytes);
        const std::uint16_t one = 0x3c00;
        std::memcpy(raw, &one, sizeof(one));
        for (std::size_t quant = 8; quant < kBlockBytes; ++quant) {
            raw[quant] = 0x11;
        }
    }
    return bytes;
}

chapter01::Tensor q5_1_tensor(const char* name,
                              std::vector<std::int64_t> shape,
                              std::size_t matrix_count) {
    return chapter01::Tensor::from_raw_bytes(
        name, std::move(shape), chapter01::DType::Q5_1,
        q5_1_ones(matrix_count));
}

}  // namespace

int main() {
    cpp_llm_engine::testing::TestSuite suite("real_moe_feed_forward");

    chapter01::ModelMetadata metadata;
    metadata.architecture = "qwen2moe";
    metadata.hidden_size = 32;
    metadata.expert_count = 2;
    metadata.expert_used_count = 1;

    chapter01::RealMoeLayerWeights weights;
    std::vector<float> router(64, 0.0F);
    router[0] = 1.0F;
    router[32] = 2.0F;
    weights.router = chapter01::Tensor::from_vector<float>(
        "router", {32, 2}, router);
    weights.shared_expert_router = chapter01::Tensor::from_vector<float>(
        "shared-router", {32}, std::vector<float>(32, 0.0F));
    weights.expert_gate = q5_1_tensor("expert-gate", {32, 32, 2}, 2);
    weights.expert_up = q5_1_tensor("expert-up", {32, 32, 2}, 2);
    weights.expert_down = q5_1_tensor("expert-down", {32, 32, 2}, 2);
    weights.shared_expert_gate =
        q5_1_tensor("shared-gate", {32, 32}, 1);
    weights.shared_expert_up = q5_1_tensor("shared-up", {32, 32}, 1);
    weights.shared_expert_down =
        q5_1_tensor("shared-down", {32, 32}, 1);

    suite.run("selected and shared experts match analytic result", [&] {
        std::vector<float> input(32, 0.0F);
        input[0] = 1.0F;
        std::vector<float> output;
        cpp_llm_engine::model::RealMoeRoute route;
        std::string error;
        CPP_LLM_REQUIRE(
            suite, cpp_llm_engine::model::run_real_moe_feed_forward(
                       metadata, weights, input, &output, &route, &error));
        CPP_LLM_REQUIRE(suite, route.expert_indices.size() == 1);
        CPP_LLM_CHECK(suite, route.expert_indices[0] == 1);
        CPP_LLM_CHECK(suite, std::fabs(route.weights[0] - 1.0F) <= 1.0e-6F);
        CPP_LLM_CHECK(
            suite, std::fabs(route.shared_expert_weight - 0.5F) <= 1.0e-6F);

        const float expert_value = 32.0F / (1.0F + std::exp(-1.0F));
        const float expected = expert_value * 1.5F;
        CPP_LLM_REQUIRE(suite, output.size() == 32);
        for (const float value : output) {
            CPP_LLM_CHECK(suite, std::fabs(value - expected) <= 1.0e-4F);
        }
    });

    suite.run("shape failure preserves prior output", [&] {
        std::vector<float> output{7.0F};
        cpp_llm_engine::model::RealMoeRoute route{{1}, {1.0F}, 0.5F};
        std::string error;
        CPP_LLM_CHECK(
            suite, !cpp_llm_engine::model::run_real_moe_feed_forward(
                       metadata, weights, std::vector<float>(31, 0.0F),
                       &output, &route, &error));
        CPP_LLM_CHECK(suite, output == std::vector<float>{7.0F});
        CPP_LLM_CHECK(suite, route.expert_indices == std::vector<std::size_t>{1});
    });

    suite.run("batch groups tokens by selected expert", [&] {
        std::vector<std::vector<float>> inputs(
            2, std::vector<float>(32, 0.0F));
        inputs[0][0] = 1.0F;
        inputs[1][0] = -1.0F;
        std::vector<std::vector<float>> expected(2);
        std::vector<cpp_llm_engine::model::RealMoeRoute> expected_routes(2);
        std::string error;
        for (std::size_t token = 0; token < inputs.size(); ++token) {
            CPP_LLM_REQUIRE(
                suite, cpp_llm_engine::model::run_real_moe_feed_forward(
                           metadata, weights, inputs[token], &expected[token],
                           &expected_routes[token], &error));
        }

        std::vector<std::vector<float>> row_wise;
        std::vector<cpp_llm_engine::model::RealMoeRoute> row_wise_routes;
        cpp_llm_engine::model::RealMoeDispatchStats row_wise_stats;
        CPP_LLM_REQUIRE(
            suite, cpp_llm_engine::model::run_real_moe_feed_forward_batch(
                       metadata, weights, inputs, &row_wise,
                       &row_wise_routes, &row_wise_stats, &error, {},
                       cpp_llm_engine::model::RealMoeBatchMode::RowWise));

        std::vector<std::vector<float>> actual;
        std::vector<cpp_llm_engine::model::RealMoeRoute> routes;
        cpp_llm_engine::model::RealMoeDispatchStats stats;
        CPP_LLM_REQUIRE(
            suite, cpp_llm_engine::model::run_real_moe_feed_forward_batch(
                       metadata, weights, inputs, &actual, &routes, &stats,
                       &error, {},
                       cpp_llm_engine::model::RealMoeBatchMode::Grouped));
        CPP_LLM_REQUIRE(suite, actual.size() == expected.size());
        CPP_LLM_CHECK(suite, row_wise_routes[0].expert_indices ==
                                 routes[0].expert_indices);
        CPP_LLM_CHECK(suite, row_wise_routes[1].expert_indices ==
                                 routes[1].expert_indices);
        CPP_LLM_CHECK(suite, row_wise_stats.assignment_count ==
                                 stats.assignment_count);
        CPP_LLM_CHECK(suite, routes[0].expert_indices ==
                                 std::vector<std::size_t>{1});
        CPP_LLM_CHECK(suite, routes[1].expert_indices ==
                                 std::vector<std::size_t>{0});
        CPP_LLM_CHECK(suite, stats.assignment_count == 2);
        CPP_LLM_CHECK(suite, stats.active_expert_count == 2);
        CPP_LLM_CHECK(suite, stats.maximum_tokens_per_expert == 1);
        for (std::size_t token = 0; token < actual.size(); ++token) {
            for (std::size_t column = 0; column < actual[token].size();
                 ++column) {
                CPP_LLM_CHECK(
                    suite, std::fabs(actual[token][column] -
                                     expected[token][column]) <= 1.0e-5F);
                CPP_LLM_CHECK(
                    suite, std::fabs(actual[token][column] -
                                     row_wise[token][column]) <= 1.0e-5F);
            }
        }
    });

    suite.run("batch failure preserves prior outputs", [&] {
        std::vector<std::vector<float>> output{{7.0F}};
        std::vector<cpp_llm_engine::model::RealMoeRoute> routes{
            {{1}, {1.0F}, 0.5F}};
        cpp_llm_engine::model::RealMoeDispatchStats stats{9, 8, 7};
        std::string error;
        CPP_LLM_CHECK(
            suite, !cpp_llm_engine::model::run_real_moe_feed_forward_batch(
                       metadata, weights,
                       {std::vector<float>(31, 0.0F)}, &output, &routes,
                       &stats, &error));
        CPP_LLM_CHECK(suite, output ==
                                 std::vector<std::vector<float>>{{7.0F}});
        CPP_LLM_CHECK(suite, routes.front().expert_indices ==
                                 std::vector<std::size_t>{1});
        CPP_LLM_CHECK(suite, stats.assignment_count == 9);
    });

    return suite.finish();
}
