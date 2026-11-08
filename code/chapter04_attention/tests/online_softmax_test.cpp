// Chapter 04：分块在线 Softmax 正确性测试。
// 职责：以三段式 CPU Attention 为 golden，核对不同 tile、因果位置和极端分数。
// 失败路径：tile_size=0 必须返回错误且不覆盖调用方输出。
// 验证：bash code/chapter04_attention/scripts/run.sh。
#include "chapter04_attention/attention_kernel.h"
#include "cpp_llm_engine/testing/test.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

namespace {

bool close(float lhs, float rhs, float tolerance = 2.0e-5F) {
    return std::abs(lhs - rhs) <= tolerance * std::max(1.0F, std::abs(rhs));
}

}  // namespace

int main() {
    cpp_llm_engine::testing::TestSuite suite("chapter04_online_softmax");

    suite.run("online tiles match the three-pass causal golden", [&] {
        constexpr std::size_t query_tokens = 3;
        constexpr std::size_t key_tokens = 5;
        constexpr std::size_t heads = 2;
        constexpr std::size_t head_dim = 3;
        constexpr std::size_t width = heads * head_dim;

        std::vector<float> query(query_tokens * width);
        std::vector<float> key(key_tokens * width);
        std::vector<float> value(key_tokens * width);
        for (std::size_t index = 0; index < query.size(); ++index) {
            query[index] = static_cast<float>((index % 7) + 1) * 0.125F;
        }
        for (std::size_t index = 0; index < key.size(); ++index) {
            key[index] = static_cast<float>(static_cast<int>(index % 11) - 5) * 0.0625F;
            value[index] = static_cast<float>((index % 13) + 1) * 0.25F;
        }

        std::vector<float> golden(query.size(), 0.0F);
        cpp_llm_engine::core::AttentionRequest request{
            query.data(), key.data(), value.data(), golden.data(),
            query_tokens, key_tokens, heads, head_dim, 1, true, 0.0F};
        CPP_LLM_REQUIRE(
            suite, chapter04_attention::cpu_reference_attention_kernel(request).is_ok());

        for (const std::size_t tile_size : {1U, 2U, 3U, 8U}) {
            std::vector<float> actual(query.size(), -1.0F);
            request.output = actual.data();
            CPP_LLM_REQUIRE(
                suite,
                chapter04_attention::cpu_online_attention_kernel(request, tile_size).is_ok());
            for (std::size_t index = 0; index < actual.size(); ++index) {
                CPP_LLM_CHECK(suite, close(actual[index], golden[index]));
            }
        }
    });

    suite.run("online rescaling remains stable for extreme scores", [&] {
        const float query[] = {1000.0F, -1000.0F};
        const float key[] = {
            1000.0F, -1000.0F,
            999.0F, -999.0F,
            -1000.0F, 1000.0F,
        };
        const float value[] = {
            1.0F, 2.0F,
            3.0F, 4.0F,
            100.0F, 200.0F,
        };
        float golden[] = {0.0F, 0.0F};
        float actual[] = {-1.0F, -1.0F};
        cpp_llm_engine::core::AttentionRequest request{
            query, key, value, golden, 1, 3, 1, 2, 0, false, 1.0F};
        CPP_LLM_REQUIRE(
            suite, chapter04_attention::cpu_reference_attention_kernel(request).is_ok());
        request.output = actual;
        CPP_LLM_REQUIRE(
            suite, chapter04_attention::cpu_online_attention_kernel(request, 1).is_ok());
        CPP_LLM_CHECK(suite, std::isfinite(actual[0]));
        CPP_LLM_CHECK(suite, std::isfinite(actual[1]));
        CPP_LLM_CHECK(suite, close(actual[0], golden[0]));
        CPP_LLM_CHECK(suite, close(actual[1], golden[1]));
    });

    suite.run("invalid tile preserves caller output", [&] {
        const float query[] = {1.0F};
        const float key[] = {1.0F};
        const float value[] = {2.0F};
        float output[] = {-7.0F};
        const cpp_llm_engine::core::AttentionRequest request{
            query, key, value, output, 1, 1, 1, 1, 0, true, 1.0F};
        const auto status =
            chapter04_attention::cpu_online_attention_kernel(request, 0);
        CPP_LLM_CHECK(suite, !status.is_ok());
        CPP_LLM_CHECK(suite, output[0] == -7.0F);
    });

    return suite.finish();
}
